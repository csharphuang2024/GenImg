import argparse
from copy import deepcopy
from datetime import datetime
import gc
import importlib
import logging
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter


PROMPT = (
    "A cute golden puppy stands on a grassy lawn, bends its legs, jumps up "
    "once with all four paws off the ground, then lands gently on the grass. "
    "Its ears bounce naturally. Full body visible, fixed camera, bright "
    "daylight, natural motion, no people, no text."
)


def setup_repository():
    repository = Path("/kaggle/working/Wan2.1-code")
    if not repository.exists():
        subprocess.run(
            ["git", "clone", "--depth", "1",
             "https://github.com/Wan-Video/Wan2.1.git", str(repository)],
            check=True,
        )
    if not (repository / "wan" / "vace.py").is_file():
        raise RuntimeError(f"Not a Wan2.1 repository: {repository}")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
         "diffusers>=0.33,<0.36", "transformers>=4.49,<5",
         "accelerate>=1.1,<2", "einops", "easydict", "ftfy", "dashscope",
         "imageio", "imageio-ffmpeg", "opencv-python-headless",
         "sentencepiece", "protobuf", "safetensors", "tqdm"],
        check=True,
    )
    sys.path.insert(0, str(repository))
    commit = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()
    print("Wan code revision:", commit, flush=True)


def find_checkpoint():
    weights = list(Path("/kaggle/input").rglob("diffusion_pytorch_model.safetensors"))
    if len(weights) != 1:
        raise RuntimeError(f"Attach one VACE Dataset; found {len(weights)} checkpoints")
    root = weights[0].parent
    for name in ("config.json", "Wan2.1_VAE.pth",
                 "models_t5_umt5-xxl-enc-bf16.pth"):
        if not (root / name).is_file():
            raise RuntimeError(f"Missing model file: {root / name}")
    tokenizer = root / "google" / "umt5-xxl"
    if not any(path.is_file() for path in tokenizer.rglob("*")):
        raise RuntimeError(f"Missing tokenizer files: {tokenizer}")
    return root


def configure_compatibility(torch):
    attention_module = importlib.import_module("wan.modules.attention")
    model_module = importlib.import_module("wan.modules.model")
    vace_module = importlib.import_module("wan.vace")
    t5_module = importlib.import_module("wan.modules.t5")

    attention_module.FLASH_ATTN_2_AVAILABLE = False
    attention_module.FLASH_ATTN_3_AVAILABLE = False

    def sdpa_attention(query=None, key=None, value=None, **kwargs):
        query = kwargs.pop("q", query)
        key = kwargs.pop("k", key)
        value = kwargs.pop("v", value)
        key_lengths = kwargs.pop("k_lens", None)
        window_size = kwargs.pop("window_size", (-1, -1))
        if window_size != (-1, -1) or kwargs:
            raise RuntimeError("This test supports unpadded, global attention only")
        if key_lengths is not None and not bool((key_lengths == key.shape[1]).all()):
            raise RuntimeError("Padded attention is not supported by this test")
        return attention_module.attention(query, key, value, dtype=torch.float16)

    model_module.flash_attention = sdpa_attention

    class LowMemoryT5(t5_module.T5EncoderModel):
        def __init__(self, text_len, dtype, device, checkpoint_path,
                     tokenizer_path, shard_fn=None):
            if shard_fn is not None or device.type != "cpu":
                raise RuntimeError("This test requires a CPU text encoder")
            self.text_len = text_len
            self.dtype = dtype
            self.device = device
            self.checkpoint_path = checkpoint_path
            self.tokenizer_path = tokenizer_path
            model = t5_module.umt5_xxl(
                encoder_only=True, dtype=dtype, device="meta"
            )
            state = torch.load(
                checkpoint_path, map_location="cpu", mmap=True, weights_only=True
            )
            model.load_state_dict(state, assign=True)
            self.model = model.eval().requires_grad_(False).to(dtype=dtype)
            self.tokenizer = t5_module.HuggingfaceTokenizer(
                name=tokenizer_path, seq_len=text_len, clean="whitespace",
                local_files_only=True,
            )

        def __call__(self, texts, device):
            started = perf_counter()
            ids, mask = self.tokenizer(
                texts, return_mask=True, add_special_tokens=True
            )
            active_columns = mask.any(dim=0).nonzero(as_tuple=True)[0]
            length = int(active_columns[-1]) + 1 if active_columns.numel() else 1
            ids = ids[:, :length].to(device)
            mask = mask[:, :length].to(device)
            lengths = mask.gt(0).sum(dim=1).long()
            print(f"Text encode START on {device}: {length}/{self.text_len} tokens",
                  flush=True)
            with torch.inference_mode():
                context = self.model(ids, mask)
            print(f"Text encode DONE in {perf_counter() - started:.1f}s", flush=True)
            return [embedding[:valid] for embedding, valid in zip(context, lengths)]

    vace_module.T5EncoderModel = LowMemoryT5
    return vace_module


def check_cuda_memory(torch):
    gc.collect()
    torch.cuda.empty_cache()
    for index in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(index)
        allocated = torch.cuda.memory_allocated(index)
        print(f"GPU {index}: free={free / 2**30:.2f}/{total / 2**30:.2f} GiB, "
              f"existing tensors={allocated / 2**30:.2f} GiB", flush=True)
        if allocated > 512 * 2**20:
            raise RuntimeError(
                "Existing GPU tensors are still alive. Restart the Python kernel "
                "before rerunning; empty_cache cannot release a referenced model."
            )


def create_pipeline(vace_module, config, checkpoint, torch):
    vae_device = torch.device("cuda:1" if torch.cuda.device_count() >= 2 else "cuda:0")
    original_vae = vace_module.WanVAE
    original_model = vace_module.VaceWanModel
    deferred_models = []

    def create_vae(*args, **kwargs):
        kwargs["device"] = vae_device
        print(f"Loading VAE directly on {vae_device}", flush=True)
        with torch.cuda.device(vae_device):
            return original_vae(*args, **kwargs)

    class CPUFirstModel:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            model = original_model.from_pretrained(*args, **kwargs)
            original_to = model.to
            had_override = "to" in model.__dict__
            deferred_models.append((model, original_to, had_override))

            def stay_on_cpu(*args, **kwargs):
                return original_to("cpu")

            model.to = stay_on_cpu
            return model

    vace_module.WanVAE = create_vae
    vace_module.VaceWanModel = CPUFirstModel
    try:
        pipeline = vace_module.WanVace(
            config=config, checkpoint_dir=str(checkpoint), t5_cpu=True
        )
    finally:
        vace_module.WanVAE = original_vae
        vace_module.VaceWanModel = original_model
        for model, original_to, had_override in deferred_models:
            if had_override:
                model.to = original_to
            else:
                del model.to
    print("Sampling model loaded on CPU; GPU 0 transfer deferred until sampling",
          flush=True)
    return pipeline


def distribute_pipeline(pipeline, torch):
    vae_device = torch.device("cuda:1" if torch.cuda.device_count() >= 2 else "cuda:0")
    pipeline.vae.model.to(vae_device)
    pipeline.vae.device = vae_device
    pipeline.vae.mean = pipeline.vae.mean.to(vae_device)
    pipeline.vae.std = pipeline.vae.std.to(vae_device)
    pipeline.vae.scale = [value.to(vae_device) for value in pipeline.vae.scale]
    torch.cuda.empty_cache()

    def move_inputs(value, device):
        if isinstance(value, torch.Tensor):
            return value.to(device)
        if isinstance(value, list):
            return [move_inputs(item, device) for item in value]
        return value

    encode_frames = pipeline.vace_encode_frames
    decode_latent = pipeline.decode_latent

    def encode_on_vae_device(frames, ref_images, masks=None, vae=None):
        active_vae = pipeline.vae if vae is None else vae
        started = perf_counter()
        print(f"VAE encode START on {active_vae.device}", flush=True)
        if active_vae.device == pipeline.device:
            pipeline.model.cpu()
            torch.cuda.empty_cache()
        with torch.cuda.device(active_vae.device):
            latents = encode_frames(
                move_inputs(frames, active_vae.device),
                move_inputs(ref_images, active_vae.device),
                masks=move_inputs(masks, active_vae.device), vae=active_vae,
            )
            latents = move_inputs(latents, pipeline.device)
            torch.cuda.synchronize(active_vae.device)
        print(f"VAE encode DONE in {perf_counter() - started:.1f}s; "
              f"sampling next on {pipeline.device}", flush=True)
        return latents

    def decode_on_vae_device(latents, ref_images=None, vae=None):
        active_vae = pipeline.vae if vae is None else vae
        started = perf_counter()
        print(f"VAE decode START on {active_vae.device}", flush=True)
        pipeline.model.cpu()
        torch.cuda.empty_cache()
        with torch.cuda.device(active_vae.device):
            videos = decode_latent(
                move_inputs(latents, active_vae.device), ref_images=ref_images,
                vae=active_vae,
            )
            torch.cuda.synchronize(active_vae.device)
        print(f"VAE decode DONE in {perf_counter() - started:.1f}s", flush=True)
        return videos

    pipeline.vace_encode_frames = encode_on_vae_device
    pipeline.decode_latent = decode_on_vae_device
    print(f"Device plan: text encoder=CPU, sampling={pipeline.device}, "
          f"VAE={vae_device}", flush=True)
    return pipeline


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--frames", type=int, default=33)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.frames < 5 or args.frames > 81 or args.frames % 4 != 1:
        parser.error("--frames must be 4n+1, between 5 and 81")
    if not 1 <= args.steps <= 100:
        parser.error("--steps must be between 1 and 100")

    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    logging.basicConfig(level=logging.INFO)
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Enable a GPU accelerator in Kaggle settings")
    for index in range(torch.cuda.device_count()):
        print(f"GPU {index}: {torch.cuda.get_device_name(index)}", flush=True)
    if torch.cuda.device_count() < 2:
        print("Only one GPU is visible. If T4 x2 is selected, restart the kernel "
              "to clear the previous CUDA_VISIBLE_DEVICES setting.", flush=True)
    check_cuda_memory(torch)
    checkpoint = find_checkpoint()
    print("Checkpoint:", checkpoint, flush=True)
    setup_repository()
    vace_module = configure_compatibility(torch)
    from wan.configs import WAN_CONFIGS
    import imageio.v2 as imageio

    config = deepcopy(WAN_CONFIGS["vace-1.3B"])
    config.param_dtype = torch.float16
    config.t5_dtype = torch.bfloat16
    print("Loading VACE; the text encoder stays on CPU", flush=True)
    started = perf_counter()
    pipeline = create_pipeline(vace_module, config, checkpoint, torch)
    distribute_pipeline(pipeline, torch)
    print(f"Pipeline loaded in {perf_counter() - started:.1f}s", flush=True)
    size = (832, 480)
    device = torch.device("cuda:0")
    frames, masks, references = pipeline.prepare_source(
        [None], [None], [None], args.frames, size, device
    )
    print("Generating:", args.prompt, flush=True)
    with torch.inference_mode():
        video = pipeline.generate(
            args.prompt, frames, masks, references, size=size,
            frame_num=args.frames, shift=16, sampling_steps=args.steps,
            guide_scale=5.0, seed=args.seed, offload_model=True,
        )
    if not bool(torch.isfinite(video).all()):
        raise RuntimeError("Non-finite output; FP16 inference was numerically unstable")
    output = Path("/kaggle/working") / (
        "puppy_jump_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".mp4"
    )
    pixels = video.detach().float().cpu().permute(1, 2, 3, 0)
    pixels = ((pixels.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).numpy()
    with imageio.get_writer(
        str(output), fps=config.sample_fps, codec="libx264", quality=8,
        macro_block_size=1,
    ) as writer:
        for frame in pixels:
            writer.append_data(frame)
    print("VIDEO_SAVED:", output, flush=True)


if __name__ == "__main__":
    main([] if "get_ipython" in globals() else None)