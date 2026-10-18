"""Greedy decoding policy with a persistent KV cache: the conversation is append-only, so each turn only
prefills the new tool result (TITO-style: the model conditions on exactly the tokens it generated)."""
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from posttrain.harness import prefix_text

BASE = os.environ.get("BASE_MODEL", "Qwen/Qwen3-0.6B")


def load(adapter=None, threads=4):
    torch.set_num_threads(threads)
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.float32)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
        model = model.merge_and_unload()
    model.eval()
    return tok, model


class QwenPolicy:
    def __init__(self, tok, model, max_new_tokens=220, max_context=7000):
        self.tok, self.model = tok, model
        self.max_new, self.max_ctx = max_new_tokens, max_context
        self.end_id = tok.convert_tokens_to_ids("<|im_end|>")
        self.gen_tokens = 0

    @torch.no_grad()
    def _feed(self, ids):
        out = self.model(input_ids=torch.tensor([ids]), past_key_values=self.cache, use_cache=True)
        self.length += len(ids)
        return out.logits[0, -1]

    @torch.no_grad()
    def _generate(self, new_ids):
        if self.length + len(new_ids) + self.max_new > self.max_ctx:
            return "<context_exhausted>"
        logits = self._feed(new_ids)
        out = []
        for _ in range(self.max_new):
            nxt = int(torch.argmax(logits))
            out.append(nxt)
            if nxt == self.end_id:
                break
            logits = self._feed([nxt])
        if out[-1] != self.end_id:          # force-close an over-long turn so the transcript stays well-formed
            self._feed([self.end_id])
        self.gen_tokens += len(out)
        return self.tok.decode(out, skip_special_tokens=True)

    def start(self, query):
        self.cache, self.length, self.gen_tokens = DynamicCache(), 0, 0
        return self._generate(self.tok(prefix_text(self.tok, query)).input_ids)

    def continue_(self, text):
        return self._generate(self.tok(text).input_ids)
