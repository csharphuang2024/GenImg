import unittest

import gen_wan_img as gv


class FakeExit(Exception):
    pass


class BuildPayloadTest(unittest.TestCase):
    def test_minimal_prompt_only(self):
        self.assertEqual(
            gv.build_payload("一只柴犬", "wan3.0-video", {}),
            {"model": "wan3.0-video", "input": {"prompt": "一只柴犬"}},
        )

    def test_negative_goes_to_input_params_stay_separate(self):
        p = gv.build_payload("a", "m", {"negative_prompt": "模糊", "size": "1280*720", "duration": 10})
        self.assertEqual(p["input"], {"prompt": "a", "negative_prompt": "模糊"})
        self.assertEqual(p["parameters"], {"size": "1280*720", "duration": 10})

    def test_parse_params_keeps_types(self):
        got = gv.parse_params(["duration=10", "prompt_extend=true", "size=1280*720"])
        self.assertEqual(got, {"duration": 10, "prompt_extend": True, "size": "1280*720"})

    def test_parse_params_rejects_bad_pair(self):
        with self.assertRaises(gv.ApiError):
            gv.parse_params(["duration"])


class BaseUrlTest(unittest.TestCase):
    def setUp(self):
        self.orig = gv.BASE_URL
        self.addCleanup(setattr, gv, "BASE_URL", self.orig)

    def test_strips_trailing_slash(self):
        gv.BASE_URL = "https://x.cn-beijing.maas.aliyuncs.com/"
        self.assertEqual(gv.base_url(), "https://x.cn-beijing.maas.aliyuncs.com")

    def test_rejects_non_http(self):
        gv.BASE_URL = "x.aliyuncs.com"
        with self.assertRaises(gv.ApiError):
            gv.base_url()

    def test_empty_falls_back_to_default(self):
        gv.BASE_URL = ""
        self.assertEqual(gv.base_url(), gv.DEFAULT_BASE_URL)

    def test_openai_compatible_suffix_is_trimmed(self):
        # 用户从控制台复制的 "OpenAI 兼容" 地址，视频接口用不了，必须剥掉
        gv.BASE_URL = "https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
        self.assertEqual(gv.base_url(), "https://token-plan.cn-beijing.maas.aliyuncs.com")

    def test_trimmed_base_yields_single_native_path(self):
        gv.BASE_URL = "https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
        url = gv.base_url() + gv.SUBMIT_PATH
        self.assertEqual(
            url,
            "https://token-plan.cn-beijing.maas.aliyuncs.com"
            "/api/v1/services/aigc/video-generation/video-synthesis",
        )
        self.assertNotIn("compatible-mode", url)

    def test_native_api_prefix_is_trimmed_too(self):
        gv.BASE_URL = "https://dashscope.aliyuncs.com/api/v1"
        self.assertEqual(gv.base_url(), "https://dashscope.aliyuncs.com")

    def test_list_models_keeps_compatible_mode_path(self):
        # 模型列表走 OpenAI 兼容协议：BASE_URL 先剥成域名，再拼回 compatible-mode
        gv.BASE_URL = "https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
        self.assertEqual(
            gv.base_url() + gv.MODELS_PATH,
            "https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1/models",
        )


class PickVideoUrlTest(unittest.TestCase):
    def test_direct_field(self):
        res = {"output": {"task_status": "SUCCEEDED", "video_url": "http://v/1.mp4"}}
        self.assertEqual(gv.pick_video_url(res), "http://v/1.mp4")

    def test_nested_results(self):
        res = {"output": {"results": [{"url": "http://v/2.mp4"}]}}
        self.assertEqual(gv.pick_video_url(res), "http://v/2.mp4")

    def test_missing_raises(self):
        with self.assertRaises(gv.ApiError):
            gv.pick_video_url({"output": {"task_status": "SUCCEEDED"}})


class WaitTaskTest(unittest.TestCase):
    def _patch(self, replies):
        calls = []

        def fake_query(task_id, api_key, insecure=False):
            calls.append(task_id)
            return replies[len(calls) - 1]

        self.orig = gv.query_task
        gv.query_task = fake_query
        self.addCleanup(setattr, gv, "query_task", self.orig)
        return calls

    def test_polls_until_succeeded(self):
        calls = self._patch([
            {"output": {"task_status": "PENDING"}},
            {"output": {"task_status": "RUNNING"}},
            {"output": {"task_status": "SUCCEEDED", "video_url": "u"}},
        ])
        res = gv.wait_task("t1", "k", interval=0)
        self.assertEqual(res["output"]["video_url"], "u")
        self.assertEqual(calls, ["t1", "t1", "t1"])

    def test_failed_raises_with_message(self):
        self._patch([{"output": {"task_status": "FAILED", "message": "绿网拦截"}}])
        with self.assertRaisesRegex(gv.ApiError, "绿网拦截"):
            gv.wait_task("t2", "k", interval=0)


class SanitizeTest(unittest.TestCase):
    def test_windows_illegal_chars(self):
        self.assertEqual(gv.sanitize('a<b>:c"d/e\\f|g?h*i'), "a_b_c_d_e_f_g_h_i")

    def test_empty_falls_back(self):
        self.assertEqual(gv.sanitize("   "), "output")

    def test_truncates(self):
        self.assertEqual(len(gv.sanitize("字" * 80, maxlen=40)), 40)


class ImagePayloadTest(unittest.TestCase):
    def test_prompt_goes_into_messages(self):
        self.assertEqual(
            gv.build_image_payload("一只柴犬", "wan2.7-image", {}),
            {
                "model": "wan2.7-image",
                "input": {"messages": [{"role": "user", "content": [{"text": "一只柴犬"}]}]},
            },
        )

    def test_size_and_n_stay_in_parameters(self):
        p = gv.build_image_payload("a", "m", {"size": "1K", "n": 2})
        self.assertEqual(p["parameters"], {"size": "1K", "n": 2})
        self.assertNotIn("negative_prompt", p["input"])

    def test_negative_prompt_goes_to_input(self):
        p = gv.build_image_payload("a", "m", {"negative_prompt": "模糊"})
        self.assertEqual(p["input"]["negative_prompt"], "模糊")
        self.assertNotIn("parameters", p)


class PickImageUrlsTest(unittest.TestCase):
    def test_single_choice(self):
        res = {"output": {"choices": [{"message": {"content": [{"image": "http://a/1.png", "type": "image"}]}}]}}
        self.assertEqual(gv.pick_image_urls(res), ["http://a/1.png"])

    def test_multiple_images_across_choices(self):
        res = {"output": {"choices": [
            {"message": {"content": [{"image": "u1"}, {"text": "skip"}]}},
            {"message": {"content": [{"image": "u2"}]}},
        ]}}
        self.assertEqual(gv.pick_image_urls(res), ["u1", "u2"])

    def test_missing_raises(self):
        with self.assertRaises(gv.ApiError):
            gv.pick_image_urls({"output": {"choices": []}})

    def test_error_code_surfaced_by_gen_image(self):
        orig = gv._request
        gv._request = lambda *a, **k: {"code": "InvalidParameter", "message": "num_images_per_prompt must be 1"}
        self.addCleanup(setattr, gv, "_request", orig)
        with self.assertRaisesRegex(gv.ApiError, "InvalidParameter"):
            gv.gen_image("a", "m", {}, "k")


if __name__ == "__main__":
    unittest.main()
