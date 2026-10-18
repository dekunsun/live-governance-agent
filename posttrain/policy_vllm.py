"""Policy backed by a vLLM OpenAI-compatible server (GPU runs).

max_new_tokens is 384 (v2.1): with 220 the SFT models' submit turn (thought + JSON card) was cut off in 14%–27% of
episodes and had to be retried.

Same interface as posttrain.policy.QwenPolicy (start / continue_ / gen_tokens / tok), but the conversation is
kept as text and re-sent on every turn; vLLM's prefix caching makes that cheap. We call /v1/completions with
the raw rendered prompt so the token stream is exactly what the CPU harness renders (no server-side chat
templating). One instance per concurrent episode: it holds the episode's text.
"""
import json
import os
import urllib.request

from transformers import AutoTokenizer

END = "<|im_end|>"


class VLLMPolicy:
    def __init__(self, base_url=None, model=None, max_new_tokens=384, max_chars=26000, temperature=0.0, timeout=600, tok=None):
        self.base = (base_url or os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8001")).rstrip("/")
        self.model = model or os.environ.get("VLLM_MODEL", "Qwen/Qwen3-4B")
        self.max_new, self.max_chars, self.temperature, self.timeout = max_new_tokens, max_chars, temperature, timeout
        self.tok = tok   # tokenizer for rendering the prompt (shared across instances; read-only)
        self.text, self.gen_tokens = "", 0

    def _complete(self):
        body = {"model": self.model, "prompt": self.text, "max_tokens": self.max_new, "temperature": self.temperature,
                "stop": [END], "skip_special_tokens": True}
        req = urllib.request.Request(self.base + "/v1/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            out = json.loads(r.read())
        ch = out["choices"][0]
        self.gen_tokens += int((out.get("usage") or {}).get("completion_tokens") or 0)
        return ch["text"]

    def _generate(self):
        if len(self.text) > self.max_chars:
            return "<context_exhausted>"
        out = self._complete()
        self.text += out + END   # close the turn exactly like the CPU policy does
        return out

    def start(self, prefix, raw=True):
        self.text, self.gen_tokens = prefix, 0
        return self._generate()

    def continue_(self, segment):
        self.text += segment
        return self._generate()


def shared_tokenizer(model):
    return AutoTokenizer.from_pretrained(model)


def wait_ready(base_url, timeout=900):
    import time
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(base_url.rstrip("/") + "/v1/models", timeout=5) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(5)
    return False
