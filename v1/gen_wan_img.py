"""万相文生图 / 文生视频：输入提示词 -> 调百炼 -> 结果保存到当前目录。

只用标准库，不需要 pip install 任何东西。

默认生成图片（wan2.7-image，同步接口）：

    python gen_video.py "一只柴犬在草地上奔跑"
    python gen_video.py                      # 不带参数则交互式输入提示词
    python gen_video.py --size 1K -o dog.png

生成视频（需该 Key 有视频模型权限）：

    python gen_video.py --mode video --model wan3.0-video --size "1280*720"

查这个 Key 能用哪些模型：python gen_video.py --list-models

API_KEY / BASE_URL 留空，请在下面 CONFIG 区填。
"""

import argparse
import json
import os
import re
import ssl
import sys
import time
import urllib.error
import urllib.request

# ---------------------------- CONFIG（填这里） ----------------------------
API_KEY = ""  # 百炼/DashScope API Key，形如 sk-xxxx

# 留空则用 DEFAULT_BASE_URL。只填到域名，不要带路径：
#   https://dashscope.aliyuncs.com  或  https://<WorkspaceId>.cn-beijing.maas.aliyuncs.com
# 注意：控制台里 "OpenAI 兼容" 那栏的 /compatible-mode/v1 不能用于视频生成，
# 万相视频只认 DashScope 原生协议；误填会被 base_url() 自动去掉。
BASE_URL = "https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com"

MODEL = "wan2.7-image"  # 图像：wan2.7-image / wan2.7-image-pro；视频：wan3.0-video
MODE = "image"  # 默认出图；--mode video 走出视频
OUTPUT_DIR = "."  # 结果保存目录，"." 即当前目录
POLL_INTERVAL = 15  # 官方建议约 15 秒轮询一次（仅视频）
TASK_TIMEOUT = 900  # 单任务最长等待秒数（仅视频）
# ------------------------------------------------------------------------

SUBMIT_PATH = "/api/v1/services/aigc/video-generation/video-synthesis"
IMAGE_PATH = "/api/v1/services/aigc/multimodal-generation/generation"  # 图像走同步接口
TASK_PATH = "/api/v1/tasks"
MODELS_PATH = "/compatible-mode/v1/models"  # 仅用于 --list-models 查模型名

# 误把 OpenAI 兼容/原生协议路径当 BASE_URL 填进来时，自动剥掉这层前缀
TRIMMABLE_SUFFIXES = ("/compatible-mode/v1", "/compatible-mode", "/api/v1", "/v1")

# 万相异步任务官方要求的标记，值必须是 enable，写成 true 会被拒
ASYNC_HEADER = ("X-DashScope-Async", "enable")

IMAGE_EXT = ".png"  # 图像接口输出格式固定 PNG
VIDEO_EXT = ".mp4"

DONE_OK = ("SUCCEEDED",)
DONE_FAIL = ("FAILED", "CANCELED", "CANCELLED", "UNKNOWN")


class ApiError(RuntimeError):
    pass


def base_url():
    base = (BASE_URL or DEFAULT_BASE_URL).strip().rstrip("/")
    for suffix in TRIMMABLE_SUFFIXES:
        if base.endswith(suffix):
            trimmed = base[: -len(suffix)].rstrip("/")
            print(f"提示: BASE_URL 末尾的 {suffix} 已忽略，视频生成走 DashScope 原生协议")
            base = trimmed
            break
    if not base.startswith(("http://", "https://")):
        raise ApiError(f"BASE_URL 必须以 http:// 或 https:// 开头，当前是 {base!r}")
    return base


def _request(url, headers, data=None, timeout=60, insecure=False):
    req = urllib.request.Request(url, data=data, headers=headers, method="POST" if data else "GET")
    ctx = ssl._create_unverified_context() if insecure else None
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
            body = resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")
        raise ApiError(f"HTTP {e.code} {url}\n{detail}") from e
    except urllib.error.URLError as e:
        raise ApiError(f"网络请求失败 {url}\n{e.reason}\n提示：公司网络有 HTTPS 代理时可加 --insecure") from e
    try:
        return json.loads(body)
    except json.JSONDecodeError as e:
        raise ApiError(f"返回不是 JSON：{body[:500]}") from e


def build_payload(prompt, model, params):
    payload = {
        "model": model,
        "input": {"prompt": prompt},
    }
    if params.get("negative_prompt"):
        payload["input"]["negative_prompt"] = params.pop("negative_prompt")
    if params:
        payload["parameters"] = params
    return payload


def build_image_payload(prompt, model, params):
    """图像接口用 messages 结构，提示词放在 content[].text。"""
    payload = {
        "model": model,
        "input": {"messages": [{"role": "user", "content": [{"text": prompt}]}]},
    }
    negative = params.pop("negative_prompt", None)
    if negative:
        payload["input"]["negative_prompt"] = negative
    if params:
        payload["parameters"] = params
    return payload


def submit_task(prompt, model, params, api_key, insecure=False):
    url = base_url() + SUBMIT_PATH
    payload = build_payload(prompt, model, params)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        ASYNC_HEADER[0]: ASYNC_HEADER[1],
    }
    print("提交任务:", url)
    print("请求体:", json.dumps(payload, ensure_ascii=False))
    res = _request(url, headers, data=json.dumps(payload).encode("utf-8"), insecure=insecure)
    task_id = (res.get("output") or {}).get("task_id")
    if not task_id:
        raise ApiError("响应里没有 task_id：" + json.dumps(res, ensure_ascii=False))
    print("task_id:", task_id, "(24 小时内有效)")
    return task_id


def gen_image(prompt, model, params, api_key, insecure=False):
    """同步出图：一次请求直接拿到图片地址，不用轮询。"""
    url = base_url() + IMAGE_PATH
    payload = build_image_payload(prompt, model, params)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    print("请求:", url)
    print("请求体:", json.dumps(payload, ensure_ascii=False))
    res = _request(url, headers, data=json.dumps(payload).encode("utf-8"), timeout=300, insecure=insecure)
    if res.get("code"):
        raise ApiError(f"{res['code']}: {res.get('message', '')}")
    return res


def query_task(task_id, api_key, insecure=False):
    url = base_url() + TASK_PATH + "/" + task_id
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    return _request(url, headers, insecure=insecure)


def list_models(api_key, insecure=False):
    """查这个 Key 实际能用的 model code，用来核对 MODEL 填得对不对。"""
    url = base_url() + MODELS_PATH
    headers = {"Authorization": f"Bearer {api_key}"}
    res = _request(url, headers, insecure=insecure)
    data = res.get("data") or res.get("models") or []
    return sorted({m["id"] for m in data if isinstance(m, dict) and m.get("id")})


def wait_task(task_id, api_key, insecure=False, timeout=TASK_TIMEOUT, interval=POLL_INTERVAL):
    deadline = time.time() + timeout
    last = ""
    while True:
        res = query_task(task_id, api_key, insecure=insecure)
        output = res.get("output") or {}
        status = str(output.get("task_status", "")).upper()
        if status != last:
            print(f"状态: {status or '未知'}")
            last = status
        if status in DONE_OK:
            return res
        if status in DONE_FAIL:
            msg = output.get("message") or res.get("message") or json.dumps(res, ensure_ascii=False)
            raise ApiError(f"任务未成功（{status}）：{msg}")
        if time.time() >= deadline:
            raise ApiError(f"等待超时（{timeout}s），task_id={task_id} 仍可用查询命令续等")
        time.sleep(interval)


def pick_image_urls(res):
    """同步出图地址在 output.choices[].message.content[].image，可能多张。"""
    output = res.get("output") or {}
    urls = []
    for choice in output.get("choices") or []:
        content = (choice.get("message") or {}).get("content") or []
        if isinstance(content, dict):
            content = [content]
        for item in content:
            if isinstance(item, dict):
                for key in ("image", "url"):
                    if isinstance(item.get(key), str) and item[key]:
                        urls.append(item[key])
    if not urls:
        raise ApiError("响应里找不到图片地址：" + json.dumps(res, ensure_ascii=False))
    return urls


def pick_video_url(res):
    """视频地址在 output.video_url，部分版本放在 output.results[0].url。"""
    output = res.get("output") or {}
    direct = output.get("video_url")
    if isinstance(direct, str) and direct:
        return direct
    results = output.get("results") or []
    if isinstance(results, dict):
        results = [results]
    for item in results:
        if isinstance(item, dict):
            for key in ("video_url", "url"):
                if isinstance(item.get(key), str) and item[key]:
                    return item[key]
    raise ApiError("响应里找不到视频地址：" + json.dumps(res, ensure_ascii=False))


def download(url, dest, insecure=False):
    print("下载:", url)
    tmp = dest + ".part"
    ctx = ssl._create_unverified_context() if insecure else None
    try:
        with urllib.request.urlopen(url, timeout=300, context=ctx) as resp:
            total = int(resp.headers.get("Content-Length") or 0)
            done = 0
            with open(tmp, "wb") as f:
                while True:
                    chunk = resp.read(1 << 16)
                    if not chunk:
                        break
                    f.write(chunk)
                    done += len(chunk)
                    if total:
                        print(f"\r  {done / 1048576:.1f} / {total / 1048576:.1f} MB", end="", flush=True)
    except urllib.error.URLError as e:
        raise ApiError(f"下载失败：{e.reason}") from e
    finally:
        if total:
            print()
    if done == 0:
        raise ApiError("下载到的文件是空的")
    os.replace(tmp, dest)
    return dest


def sanitize(stem, maxlen=40):
    stem = re.sub(r'[<>:"/\\|?*\x00-\x1f\s]+', "_", stem).strip("_")
    return (stem[:maxlen] or "output")


def parse_params(pairs):
    """--param key=value，值尽量按 JSON 解析，便于传 duration/seed 这类数字参数。"""
    out = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise ApiError(f"--param 需要写成 key=value，收到 {pair!r}")
        k, v = pair.split("=", 1)
        try:
            out[k.strip()] = json.loads(v)
        except json.JSONDecodeError:
            out[k.strip()] = v
    return out


def build_arg_parser():
    p = argparse.ArgumentParser(description="万相文生图（默认）/ 文生视频，结果保存到当前目录")
    p.add_argument("prompt", nargs="?", help="提示词；省略则交互输入")
    p.add_argument("--mode", choices=("image", "video"), default=None,
                   help=f"image 出图 / video 出片，默认用代码里的 {MODE}")
    p.add_argument("--model", default=None, help=f"模型名，默认用代码里的 {MODEL}")
    p.add_argument("--api-key", default=None, help="覆盖代码里的 API_KEY")
    p.add_argument("--base-url", default=None, help="覆盖代码里的 BASE_URL")
    p.add_argument("--size", default=None, help='图像可填 1K/2K 或 "2048*2048"；视频填 "1280*720"（含 * 要用引号）')
    p.add_argument("--n", type=int, default=None, help="图像张数 1-4，默认 1")
    p.add_argument("--duration", type=int, default=None, help="视频时长（秒）")
    p.add_argument("--negative", default=None, help="反向提示词")
    p.add_argument("--param", action="append", metavar="K=V", help="额外参数，可重复；数字会自动转成 JSON 类型")
    p.add_argument("-o", "--output", default=None, help="输出文件名，默认 时间戳_提示词前缀.png/.mp4")
    p.add_argument("--timeout", type=int, default=TASK_TIMEOUT, help="视频最长等待秒数")
    p.add_argument("--interval", type=int, default=POLL_INTERVAL, help="视频轮询间隔秒数")
    p.add_argument("--insecure", action="store_true", help="跳过 HTTPS 证书校验（仅用于本机代理证书问题）")
    p.add_argument("--list-models", nargs="?", const="", default=None, metavar="FILTER",
                   help="只列出该 Key 可用的 model code 后退出，可带过滤词如 wan")
    return p


def dest_name(args, mode, index=0, total=1):
    ext = IMAGE_EXT if mode == "image" else VIDEO_EXT
    stem = args.output or f"{time.strftime('%Y%m%d-%H%M%S')}_{sanitize(args.prompt)}"
    if args.output and total > 1:
        head, tail = os.path.splitext(stem)
        stem = f"{head}_{index + 1}{tail or ext}"
    if not stem.lower().endswith(ext):
        stem += ext
    return os.path.join(OUTPUT_DIR, stem)


def main(argv=None):
    args = build_arg_parser().parse_args(argv)

    api_key = args.api_key or API_KEY
    if args.base_url:
        global BASE_URL
        BASE_URL = args.base_url

    if args.list_models is not None:
        if not api_key:
            print("请先在本文件 CONFIG 区填 API_KEY（或用 --api-key 传入）", file=sys.stderr)
            return 2
        try:
            ids = list_models(api_key, args.insecure)
        except ApiError as e:
            print("失败:", e, file=sys.stderr)
            return 1
        kw = args.list_models.lower()
        hit = [i for i in ids if not kw or kw in i.lower()]
        print(f"该 Key 可用模型 {len(ids)} 个，匹配 {len(hit)} 个：")
        for i in hit:
            print("  ", i)
        return 0

    args.prompt = args.prompt or input("提示词: ").strip()
    if not args.prompt:
        print("没有提示词，退出")
        return 1

    if not api_key:
        print("请先在本文件 CONFIG 区填 API_KEY（或用 --api-key 传入）", file=sys.stderr)
        return 2

    mode = args.mode or MODE
    model = args.model or MODEL
    params = parse_params(args.param)
    if args.size:
        params["size"] = args.size
    if args.n is not None:
        params["n"] = args.n
    if args.duration is not None:
        params["duration"] = args.duration
    if args.negative:
        params["negative_prompt"] = args.negative

    saved = []
    try:
        if mode == "image":
            res = gen_image(args.prompt, model, params, api_key, args.insecure)
            urls = pick_image_urls(res)
            if len(urls) > 1:
                print(f"返回 {len(urls)} 张图")
            for i, url in enumerate(urls):
                path = download(url, dest_name(args, mode, i, len(urls)), args.insecure)
                saved.append(path)
        else:
            task_id = submit_task(args.prompt, model, params, api_key, args.insecure)
            res = wait_task(task_id, api_key, args.insecure, args.timeout, args.interval)
            path = download(pick_video_url(res), dest_name(args, mode), args.insecure)
            saved.append(path)
    except ApiError as e:
        print("失败:", e, file=sys.stderr)
        if "not exist" in str(e).lower():
            print(f"提示: MODEL={model} 在该 Key 下不可用，"
                  f"先跑 python gen_video.py --list-models wan 查可用模型名", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\n已中断", file=sys.stderr)
        return 130

    for p in saved:
        print(f"已保存 {p} ({os.path.getsize(p) / 1048576:.2f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
