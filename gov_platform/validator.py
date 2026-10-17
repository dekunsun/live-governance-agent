"""Reference validator: every id the agent cites must exist on the platform or in this run's tool results."""
import json
import re

from .tools import ID_PATTERNS

MODEL_VOCAB = {"model_drift_threshold", "model_update", "model_version", "model_versions", "model_calls", "model_text",
               "model_refusal", "model_truncated", "model_changes", "model_id"}


class Validator:
    def __init__(self, platform):
        self.p = platform

    def extract(self, text):
        found = {}
        for kind, pat in ID_PATTERNS.items():
            ids = set(re.findall(r"(?<![A-Za-z0-9_-])" + pat + r"(?![A-Za-z0-9_-])", text))
            if ids:
                found[kind] = ids
        return found

    def check_text(self, text, run_refs=None):
        """Return list of (kind, id) that do not exist."""
        unknown = []
        for kind, ids in self.extract(text).items():
            if kind == "ref_id":
                known = set((run_refs if run_refs is not None else self.p.refs).keys())
            else:
                known = self.p.known.get(kind, set())
            for i in sorted(ids):
                if kind == "model" and i in MODEL_VOCAB:
                    continue
                if i not in known:
                    unknown.append((kind, i))
        return unknown

    def check_card(self, card, run_refs=None):
        """Validate an attribution/health card: ids in text + structured fields (metric, strategy_ids, change_ids)."""
        text = json.dumps(card, ensure_ascii=False)
        unknown = self.check_text(text, run_refs)
        pc = card.get("primary_cause") or {}
        for sid in pc.get("strategy_ids", []) or []:
            if sid not in self.p.known["strategy_id"] and ("strategy_id", sid) not in unknown:
                unknown.append(("strategy_id", sid))
        for cid in pc.get("change_ids", []) or []:
            if cid not in self.p.known["change_id"] and ("change_id", cid) not in unknown:
                unknown.append(("change_id", cid))
        for gid in pc.get("guild_ids", []) or []:
            if gid not in self.p.known["guild_id"] and ("guild_id", gid) not in unknown:
                unknown.append(("guild_id", gid))
        m = (card.get("metric") or {}).get("key")
        if m and m not in self.p.known["metric"]:
            unknown.append(("metric", m))
        for ev in card.get("evidence", []) or []:
            if ev.get("metric") and ev["metric"] not in self.p.known["metric"]:
                unknown.append(("metric", ev["metric"]))
        return unknown
