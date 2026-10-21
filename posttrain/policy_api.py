"""Policy backed by an external chat + tool-calling API, for the same harness the local models use.

The local policies (posttrain/policy.py, posttrain/policy_vllm.py) speak raw completions: the harness renders the
conversation as text and parses `<tool_call>{...}</tool_call>` out of what the model generates. External models are
served through conversation endpoints with structured tool calls instead. This adapter keeps the harness untouched:

  * `start(prefix, raw=True)` receives what `harness_v2.prefix_text` rendered through `policy.tok`. Our `tok` is a
    shim whose `apply_chat_template` returns the system prompt, user turn and tool list as JSON instead of Qwen
    chat markup, so the adapter gets the original messages back without re-parsing text.
  * each turn the model's structured tool call is rendered back into the `<tool_call>` text the harness parses;
    the harness's normalisation, gates (traffic first, ids must have appeared, cause enum) and the grader all run
    unchanged. Tool results, gate rejections and format errors come back through `continue_(obs_segment)` and are
    appended as tool_result / user messages.
  * one tool per turn, as the harness assumes: parallel tool use is disabled where the API allows; if a model still
    returns several calls only the first is executed and the others get a "not executed" tool result.

Providers:
  anthropic   Anthropic Messages API (official `anthropic` SDK). Claude Sonnet 5.5 / Opus 5.5: thinking is on by
              default and cannot be disabled, `temperature` is rejected, `tool_choice` cannot force a tool
              (we use auto + the prompt's "must end with submit_card"). Prompt caching on the system block and on
              the growing conversation. Optional Message Batches mode (50% price): a BatchBroker collects the next
              request of every live episode, submits one batch per round, and hands results back.
  openrouter  OpenAI-compatible chat completions through OpenRouter (official `openai` SDK), greedy decoding.

Usage tracking: every policy instance accumulates token usage and an estimated cost in USD (`policy.usage`,
`policy.cost_usd`) from the price table below (OpenRouter's own per-request cost is used when it reports one).
Keys come from the environment only (ANTHROPIC_API_KEY / OPENROUTER_API_KEY); they are never logged.
"""
import copy
import json
import os
import re
import threading
import time
import uuid

MARK = "\x00API_POLICY_MESSAGES\x00"

# USD per 1M tokens. Anthropic: input / output / cache read / cache write (5 min) / cache write (1 h).
PRICES = {
    "claude-sonnet-5-5": {"in": 2.0, "out": 10.0, "cache_read": 0.20, "cache_w5m": 2.5, "cache_w1h": 4.0},
    "claude-opus-5-5": {"in": 4.0, "out": 20.0, "cache_read": 0.20, "cache_w5m": 5.0, "cache_w1h": 8.0},
    "claude-haiku-5-5": {"in": 0.10, "out": 0.50, "cache_read": 0.01, "cache_w5m": 0.125, "cache_w1h": 0.20},
    # OpenRouter list prices (2026-10); the per-request `usage.cost` OpenRouter returns overrides these.
    "qwen/qwen3.5-397b-a17b": {"in": 0.39, "out": 2.34, "cache_read": 0.39},
    "qwen/qwen3.5-122b-a10b": {"in": 0.26, "out": 2.08, "cache_read": 0.26},
}
BATCH_DISCOUNT = 0.5


def clone_platform(platform):
    """A per-episode view of a world: the data frames are shared read-only, the per-run ref table is private."""
    p = copy.copy(platform)
    p.refs, p._ref_n = {}, 0
    return p


class ChatShim:
    """Stands in for a tokenizer: harness_v2.prefix_text calls `tok.apply_chat_template(msgs, tools=...)`."""

    def apply_chat_template(self, msgs, tools=None, tokenize=False, add_generation_prompt=True, **kw):
        return MARK + json.dumps({"messages": msgs, "tools": tools or []}, ensure_ascii=False)


OBS_RE = re.compile(r"<tool_response>\n(.*)\n</tool_response>", re.S)
NOT_EXECUTED = "每轮只能调用一个工具：这个调用没有执行。需要的话下一轮再单独调用。"


def _zero():
    return {"input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0, "requests": 0}


class APIPolicy:
    """Same interface as VLLMPolicy: start / continue_ / gen_tokens / tok. One instance per episode."""

    def __init__(self, provider, model, client=None, fewshot=None, effort=None, max_tokens=None, temperature=0.0,
                 max_turn_chars=60000, broker=None, cache_ttl=None, reasoning=None, timeout=300, max_retries=5):
        self.provider, self.model = provider, model
        self.tok = ChatShim()
        self.fewshot, self.effort, self.temperature = fewshot, effort, temperature
        self.max_tokens = max_tokens or (8000 if provider == "anthropic" else 4096)
        self.max_turn_chars, self.broker, self.cache_ttl, self.reasoning = max_turn_chars, broker, cache_ttl, reasoning
        self.client = client or self._make_client(timeout, max_retries)
        self.price = PRICES.get(model, {"in": 0.0, "out": 0.0, "cache_read": 0.0})
        self.system, self.tools, self.messages, self.pending = "", [], [], []
        self.gen_tokens, self.usage, self.cost_usd, self.stop_reasons = 0, _zero(), 0.0, []

    def _make_client(self, timeout, max_retries):
        if self.provider == "anthropic":
            import anthropic
            return anthropic.Anthropic(timeout=timeout, max_retries=max_retries)
        if self.provider == "openrouter":
            from openai import OpenAI
            return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"],
                          timeout=timeout, max_retries=max_retries)
        raise ValueError(self.provider)

    # ------------------------------------------------------------------ harness interface
    def start(self, prefix, raw=True):
        if prefix.startswith(MARK):
            d = json.loads(prefix[len(MARK):])
            msgs, tools = d["messages"], d["tools"]
            system = "\n\n".join(m["content"] for m in msgs if m["role"] == "system")
            user = "\n\n".join(m["content"] for m in msgs if m["role"] == "user")
        else:   # harness v1 passes the bare query
            from posttrain.harness import SYSTEM, TOOLS
            system, user, tools = SYSTEM, prefix, TOOLS
        self.system = system + ("\n\n" + self.fewshot if self.fewshot else "")
        self.tools = [self._tool(t) for t in tools]
        self.messages, self.pending = [{"role": "user", "content": user}], []
        self.gen_tokens, self.usage, self.cost_usd, self.stop_reasons = 0, _zero(), 0.0, []
        return self._turn()

    def continue_(self, segment):
        m = OBS_RE.search(segment)
        obs = m.group(1) if m else segment.strip()
        if self.pending:
            self.messages.append(self._tool_results(obs))
        else:   # no tool call last turn (format error from the harness): plain user message
            self.messages.append({"role": "user", "content": obs})
        self.pending = []
        return self._turn()

    # ------------------------------------------------------------------ provider plumbing
    def _tool(self, t):
        fn = t["function"] if t.get("type") == "function" else t
        if self.provider == "anthropic":
            return {"name": fn["name"], "description": fn.get("description", ""), "input_schema": fn["parameters"]}
        return {"type": "function", "function": {"name": fn["name"], "description": fn.get("description", ""), "parameters": fn["parameters"]}}

    def _tool_results(self, obs):
        if self.provider == "anthropic":
            blocks = [{"type": "tool_result", "tool_use_id": self.pending[0], "content": obs}]
            blocks += [{"type": "tool_result", "tool_use_id": pid, "content": NOT_EXECUTED} for pid in self.pending[1:]]
            return {"role": "user", "content": blocks}
        # OpenAI-compatible: one `tool` message per call; the caller appends them in order
        return [{"role": "tool", "tool_call_id": self.pending[0], "content": obs}] + \
               [{"role": "tool", "tool_call_id": pid, "content": NOT_EXECUTED} for pid in self.pending[1:]]

    def _turn(self):
        if len(json.dumps(self.messages, ensure_ascii=False, default=str)) > self.max_turn_chars:
            return "<context_exhausted>"
        if self.provider == "anthropic":
            text, calls, assistant, usage, stop = self._anthropic()
        else:
            text, calls, assistant, usage, stop = self._openrouter()
        self.messages.append(assistant)
        self.stop_reasons.append(stop)
        self.pending = [c[0] for c in calls]
        out = text.strip()
        if calls:
            _, name, args = calls[0]
            out = (out + "\n" if out else "") + "<tool_call>\n" + json.dumps({"name": name, "arguments": args}, ensure_ascii=False) + "\n</tool_call>"
        return out

    def _account(self, usage, cost=None):
        for k in self.usage:
            if k != "requests":
                self.usage[k] += int(usage.get(k) or 0)
        self.usage["requests"] += 1
        self.gen_tokens += int(usage.get("output_tokens") or 0)
        if cost is None:
            p = self.price
            w = p.get("cache_w1h" if self.cache_ttl == "1h" else "cache_w5m", 0.0)
            cost = (usage.get("input_tokens", 0) * p["in"] + usage.get("output_tokens", 0) * p["out"]
                    + usage.get("cache_read_input_tokens", 0) * p.get("cache_read", 0.0)
                    + usage.get("cache_creation_input_tokens", 0) * w) / 1e6
            if self.broker is not None:
                cost *= BATCH_DISCOUNT
        self.cost_usd += cost

    # ---- Anthropic Messages API
    def _anthropic_params(self):
        cc = {"type": "ephemeral", **({"ttl": self.cache_ttl} if self.cache_ttl else {})}
        msgs = copy.deepcopy(self.messages)
        last = msgs[-1]   # cache the whole conversation so far: the next turn only pays for the new tool result
        if isinstance(last["content"], str):
            last["content"] = [{"type": "text", "text": last["content"], "cache_control": cc}]
        else:
            last["content"][-1]["cache_control"] = cc
        params = {"model": self.model, "max_tokens": self.max_tokens,
                  "system": [{"type": "text", "text": self.system, "cache_control": cc}],
                  "tools": self.tools, "tool_choice": {"type": "auto", "disable_parallel_tool_use": True}, "messages": msgs}
        if self.effort:
            params["output_config"] = {"effort": self.effort}
        return params

    def _anthropic(self):
        params = self._anthropic_params()
        resp = self.broker.call(params) if self.broker is not None else self.client.messages.create(**params)
        d = resp.model_dump(exclude_none=True) if hasattr(resp, "model_dump") else resp
        content = d.get("content") or []
        text = "".join(b.get("text", "") for b in content if b.get("type") == "text")
        calls = [(b["id"], b["name"], b.get("input") or {}) for b in content if b.get("type") == "tool_use"]
        if not content:   # the API rejects an empty assistant turn on the next request
            content = [{"type": "text", "text": "（空回复）"}]
        u = d.get("usage") or {}
        self._account({"input_tokens": u.get("input_tokens", 0), "output_tokens": u.get("output_tokens", 0),
                       "cache_read_input_tokens": u.get("cache_read_input_tokens", 0),
                       "cache_creation_input_tokens": u.get("cache_creation_input_tokens", 0)})
        stop = d.get("stop_reason")
        if stop == "refusal":
            text = text or f"（模型拒绝：{(d.get('stop_details') or {}).get('category')}）"
        return text, calls, {"role": "assistant", "content": content}, u, stop

    # ---- OpenRouter (OpenAI-compatible)
    def _openrouter(self):
        msgs = [{"role": "system", "content": self.system}]
        for m in self.messages:
            msgs.extend(m if isinstance(m, list) else [m])
        extra = {"usage": {"include": True}}
        if self.reasoning is not None:
            extra["reasoning"] = self.reasoning
        resp = self.client.chat.completions.create(model=self.model, messages=msgs, tools=self.tools, tool_choice="auto",
                                                   parallel_tool_calls=False, temperature=self.temperature,
                                                   max_tokens=self.max_tokens, extra_body=extra)
        d = resp.model_dump(exclude_none=True) if hasattr(resp, "model_dump") else resp
        ch = (d.get("choices") or [{}])[0]
        msg = ch.get("message") or {}
        text = msg.get("content") or ""
        calls = []
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function") or {}
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except Exception:
                args = {"_raw": fn.get("arguments")}
            calls.append((tc.get("id") or f"call_{uuid.uuid4().hex[:8]}", fn.get("name"), args if isinstance(args, dict) else {}))
        assistant = {"role": "assistant", "content": text}
        if msg.get("tool_calls"):
            assistant["tool_calls"] = [{"id": cid, "type": "function", "function": {"name": tc["function"]["name"], "arguments": tc["function"].get("arguments") or "{}"}}
                                       for cid, tc in zip([c[0] for c in calls], msg["tool_calls"])]
        if msg.get("reasoning_details"):   # OpenRouter: pass reasoning back unchanged for interleaved thinking
            assistant["reasoning_details"] = msg["reasoning_details"]
        u = d.get("usage") or {}
        cached = ((u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
        usage = {"input_tokens": (u.get("prompt_tokens") or 0) - cached, "output_tokens": u.get("completion_tokens") or 0,
                 "cache_read_input_tokens": cached, "cache_creation_input_tokens": 0}
        self._account(usage, cost=u.get("cost"))
        return text, calls, assistant, u, ch.get("finish_reason")


# ---------------------------------------------------------------------- Message Batches
class BatchBroker:
    """Turns the per-episode blocking `call(params)` into one Message Batch per round.

    Every live episode (thread) posts its next request and blocks; when all live episodes have posted (or the oldest
    request has waited `max_wait` seconds, to tolerate a straggler) the broker submits a batch, polls it, and wakes
    every thread with its result. Episodes register/unregister so the broker knows how many to wait for.
    """

    def __init__(self, client, poll=20, max_wait=90, log=print):
        self.client, self.poll, self.max_wait, self.log = client, poll, max_wait, log
        self.cv = threading.Condition()
        self.active, self.pending, self.results, self.first_post = 0, {}, {}, None
        self.rounds, self.closed = 0, False
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def register(self):
        with self.cv:
            self.active += 1

    def unregister(self):
        with self.cv:
            self.active -= 1
            self.cv.notify_all()

    def close(self):
        with self.cv:
            self.closed = True
            self.cv.notify_all()

    def call(self, params):
        cid = uuid.uuid4().hex[:24]
        with self.cv:
            self.pending[cid] = params
            if self.first_post is None:
                self.first_post = time.time()
            self.cv.notify_all()
            while cid not in self.results:
                self.cv.wait(5)
            r = self.results.pop(cid)
        if isinstance(r, Exception):
            raise r
        return r

    def _ready(self):
        if not self.pending:
            return False
        return len(self.pending) >= self.active or (time.time() - self.first_post) > self.max_wait

    def _loop(self):
        while True:
            with self.cv:
                while not self._ready() and not self.closed:
                    self.cv.wait(2)
                if self.closed and not self.pending:
                    return
                batch, self.pending, self.first_post = self.pending, {}, None
            try:
                results = self._run(batch)
            except Exception as e:   # submission failed as a whole: every caller gets the error
                results = {cid: e for cid in batch}
            with self.cv:
                self.results.update(results)
                self.cv.notify_all()

    def _run(self, batch):
        from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
        from anthropic.types.messages.batch_create_params import Request
        self.rounds += 1
        t0 = time.time()
        b = self.client.messages.batches.create(requests=[Request(custom_id=cid, params=MessageCreateParamsNonStreaming(**p)) for cid, p in batch.items()])
        self.log(f"[batch] round {self.rounds}: {len(batch)} requests submitted as {b.id}")
        while True:
            b = self.client.messages.batches.retrieve(b.id)
            if b.processing_status == "ended":
                break
            time.sleep(self.poll)
        out = {}
        for r in self.client.messages.batches.results(b.id):
            if r.result.type == "succeeded":
                out[r.custom_id] = r.result.message
            else:
                err = getattr(r.result, "error", None)
                out[r.custom_id] = RuntimeError(f"batch {r.result.type}: {getattr(err, 'message', err)}")
        for cid in batch:
            out.setdefault(cid, RuntimeError("batch result missing"))
        self.log(f"[batch] round {self.rounds}: ended in {time.time() - t0:.0f}s, {b.request_counts.succeeded} ok / {b.request_counts.errored} errored")
        return out
