"""Simulated governance-platform tools.

Every tool is a method on ``Platform``. Inputs are validated; unknown ids or metrics raise
``ToolError`` (the agent must report missing data instead of inventing it). Every result
carries a ``ref_id`` (Q-0001, ...) so the agent can cite it as evidence, and the
``Validator`` can later confirm that every citation points at something real.

Only ``draft_change`` writes, and it writes a draft file that nothing else reads.
"""
import json
import os
from datetime import date, timedelta

import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "out")
DRAFT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "drafts")


class ToolError(Exception):
    def __init__(self, code, message, hint=None):
        super().__init__(message)
        self.code, self.message, self.hint = code, message, hint

    def to_dict(self):
        return {"error": self.code, "message": self.message, "hint": self.hint}


# ----------------------------------------------------------------------------- metric registry
METRICS = {
    "sessions": {"zh": "开播场次", "level": "market", "kind": "count", "version": "v1",
                 "definition": "窗口内该市场的直播 session 数（流量分母）。与场景无关。"},
    "hits": {"zh": "策略命中数", "level": "strategy", "kind": "count", "version": "v1",
             "definition": "策略命中次数。一个 session 可被多条策略命中，按命中次数计。"},
    "manual_reviews": {"zh": "转人审数", "level": "strategy", "kind": "count", "version": "v1",
                       "definition": "动作为 manual_review 的命中次数。"},
    "enforcements": {"zh": "处罚 session 数", "level": "strategy", "kind": "count", "version": "v2",
                     "definition": "被处罚的 session 数（limit_traffic/warn/suspend_live/ban_7d）。每 session 计一次，"
                                   "归属到严重度最高的策略（primary_strategy）。manual_review 和 observe 不算处罚。"},
    "appeals": {"zh": "申诉数", "level": "strategy", "kind": "count", "version": "v1",
                "definition": "对处罚提交的申诉数，按申诉提交日期归属。"},
    "overturns": {"zh": "申诉推翻数", "level": "strategy", "kind": "count", "version": "v1",
                  "definition": "申诉裁决为推翻（误判）的数量，按裁决日期归属。"},
    "reports": {"zh": "用户举报数", "level": "scenario", "kind": "count", "version": "v1",
                "definition": "用户举报数，按举报原因归到场景。与策略是否命中无关，可作为召回侧的独立信号。"},
    "hit_rate": {"zh": "命中率", "level": "strategy", "kind": "ratio", "version": "v1",
                 "num": "hits", "den": "sessions", "definition": "hits / sessions。"},
    "enforcement_rate": {"zh": "处罚率", "level": "strategy", "kind": "ratio", "version": "v2",
                         "num": "enforcements", "den": "sessions", "definition": "enforcements / sessions。"},
    "appeal_rate": {"zh": "申诉率", "level": "strategy", "kind": "ratio", "version": "v1",
                    "num": "appeals", "den": "enforcements", "definition": "appeals / enforcements。"},
    "overturn_rate": {"zh": "申诉推翻率", "level": "strategy", "kind": "ratio", "version": "v2",
                      "num": "overturns", "den": "appeals", "definition": "overturns / appeals（分母是申诉数）。"},
    "overturn_per_enforcement": {"zh": "每处罚推翻率", "level": "strategy", "kind": "ratio", "version": "v1",
                                 "num": "overturns", "den": "enforcements",
                                 "definition": "overturns / enforcements（分母是处罚数）。与 overturn_rate 分母不同，口径冲突时需明确。"},
    "precision_est": {"zh": "抽检精确率", "level": "strategy", "kind": "estimate", "version": "v3",
                      "definition": "QA 抽检的命中样本中确认违规的比例（每策略/市场/天最多抽 10 条）。样本不足时不可靠。"},
    "recall_est": {"zh": "召回估计", "level": "scenario", "kind": "estimate", "version": "v2",
                   "definition": "分层估计：TP_est = 场景命中 session 数 × precision_est；漏放估计 = 未命中 QA 样本中该场景违规数按分层权重放大；"
                                 "recall_est = TP_est / (TP_est + 漏放估计)。小市场样本少，仅看趋势。"},
}
ALIASES = {
    "误杀率": ["overturn_rate", "overturn_per_enforcement", "precision_est"],
    "误杀申诉": ["overturns"], "误杀": ["overturns"], "申诉推翻": ["overturns"], "推翻": ["overturns"],
    "申诉": ["appeals"], "处罚": ["enforcements"], "处罚量": ["enforcements"], "命中": ["hits"],
    "人审": ["manual_reviews"], "举报": ["reports"], "召回": ["recall_est"], "精确率": ["precision_est"],
    "准确率": ["precision_est"], "流量": ["sessions"], "开播": ["sessions"], "命中率": ["hit_rate"], "处罚率": ["enforcement_rate"],
}
ID_PATTERNS = {
    "strategy_id": r"S-\d{3}", "change_id": r"C-\d{3}", "session_id": r"LS-\d{7}", "hit_id": r"HT-\d{7}",
    "appeal_id": r"AP-\d{6}", "enforcement_id": r"EN-\d{7}", "qa_id": r"QA-\d{6}", "policy_id": r"P-\d{1,2}",
    "model": r"model_[a-z_]+", "ref_id": r"Q-\d{4}",
}


def _d(s):
    return date.fromisoformat(s)


def _psi(base, cur, bins=10):
    edges = np.quantile(base, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -1e-9, 1 + 1e-9
    b = np.histogram(base, edges)[0] / max(len(base), 1) + 1e-6
    c = np.histogram(cur, edges)[0] / max(len(cur), 1) + 1e-6
    return float(np.sum((c - b) * np.log(c / b)))


class Platform:
    def __init__(self, data_dir=DATA_DIR, user="demo.user", ref_prefix="Q"):
        self.data_dir, self.user = data_dir, user
        rd = lambda n: pd.read_csv(os.path.join(data_dir, n))
        self.sessions = rd("sessions.csv")
        self.hits = rd("hit_logs.csv")
        self.enforcements = rd("enforcements.csv")
        self.appeals = rd("appeals.csv")
        self.qa = rd("qa_samples.csv")
        self.reports = rd("reports.csv")
        self.metrics = rd("metrics_daily.csv")
        self.metrics_strategy = rd("metrics_daily_strategy.csv")
        self.strategies = json.load(open(os.path.join(data_dir, "strategies.json")))
        self.change_log = json.load(open(os.path.join(data_dir, "change_log.json")))
        self.guilds = json.load(open(os.path.join(data_dir, "guilds.json")))
        self.guild_by_id = {g["guild_id"]: g for g in self.guilds}
        self.meta = json.load(open(os.path.join(data_dir, "meta.json")))
        self.by_id = {s["strategy_id"]: s for s in self.strategies}
        self.sess_idx = self.sessions.set_index("session_id")
        self.known = {
            "strategy_id": set(self.by_id), "change_id": {c["change_id"] for c in self.change_log},
            "session_id": set(self.sessions.session_id), "hit_id": set(self.hits.hit_id),
            "appeal_id": set(self.appeals.appeal_id), "enforcement_id": set(self.enforcements.enforcement_id),
            "qa_id": set(self.qa.qa_id), "policy_id": {p["policy_id"] for p in self.meta["policies"]},
            "model": {m["model"] for m in self.meta["model_versions"]}, "metric": set(METRICS), "guild_id": set(self.guild_by_id),
            "market": set(self.meta["markets"]), "scenario": set(self.meta["scenarios"]),
        }
        self.refs = {}          # ref_id -> tool name (per run)
        self._ref_n = 0
        self._ref_prefix = ref_prefix

    # ------------------------------------------------------------------ helpers
    def _ref(self, tool):
        self._ref_n += 1
        rid = f"{self._ref_prefix}-{self._ref_n:04d}"
        self.refs[rid] = tool
        return rid

    def _check_market(self, m):
        if m is None or m == "ALL":
            return None
        ms = [m] if isinstance(m, str) else list(m)
        bad = [x for x in ms if x not in self.known["market"]]
        if bad:
            raise ToolError("unknown_market", f"未知市场 {bad}", f"可用市场: {sorted(self.known['market'])}")
        return ms

    def _check_scenario(self, s):
        if s is None:
            return None
        if s not in self.known["scenario"]:
            raise ToolError("unknown_scenario", f"未知场景 {s}", f"可用场景: {self.meta['scenarios']}")
        return s

    def _check_strategy(self, sid):
        if sid is None:
            return None
        if sid not in self.by_id:
            raise ToolError("unknown_strategy", f"策略 {sid} 不存在", "用 get_strategy_config 查看策略列表；不要猜测策略编号")
        return sid

    def _check_window(self, start, end):
        try:
            s, e = _d(start), _d(end)
        except Exception:
            raise ToolError("bad_date", f"日期格式错误: {start}/{end}", "使用 YYYY-MM-DD")
        lo, hi = _d(self.meta["start"]), _d(self.meta["end"])
        if e < s:
            raise ToolError("bad_window", "end_date 早于 start_date")
        if e < lo or s > hi:
            raise ToolError("no_data", f"窗口 {start}~{end} 无数据", f"数据范围 {self.meta['start']} ~ {self.meta['end']}")
        clipped = s < lo or e > hi
        return max(s, lo).isoformat(), min(e, hi).isoformat(), clipped

    def _check_guild(self, g):
        if g is None:
            return None
        if g not in self.guild_by_id:
            raise ToolError("unknown_guild", f"公会 {g} 不存在", "用 get_guild_info 查看公会；不要猜测公会编号")
        return g

    def _with_guild(self, df):
        return df.assign(guild_id=df.session_id.map(self.sess_idx.guild_id).fillna(""))

    def resolve_metric(self, name):
        """Return (metric_key or None, candidates). Chinese aliases may be ambiguous (口径冲突)."""
        if name in METRICS:
            return name, [name]
        for k, v in ALIASES.items():
            if k in name:
                return (v[0], v) if len(v) == 1 else (None, v)
        return None, []

    # ------------------------------------------------------------------ tools
    def query_metrics(self, metric, start_date, end_date, market=None, scenario=None, strategy_id=None,
                      group_by=None, granularity="day", guild_id=None):
        """Query governance metrics by day or as a window total, grouped by market/scenario/strategy_id."""
        key, cands = self.resolve_metric(metric)
        if key is None:
            if cands:
                raise ToolError("ambiguous_metric", f"“{metric}” 对应多个口径: {cands}",
                                "口径冲突：请选定一个指标名重新查询，并在结论中说明分母。 " +
                                "; ".join(f"{c}: {METRICS[c]['definition']}" for c in cands))
            raise ToolError("unknown_metric", f"未知指标 {metric}", f"可用指标: {sorted(METRICS)}")
        start, end, clipped = self._check_window(start_date, end_date)
        markets = self._check_market(market)
        scenario = self._check_scenario(scenario)
        strategy_id = self._check_strategy(strategy_id)
        guild_id = self._check_guild(guild_id)
        group_by = list(group_by or [])
        for g in group_by:
            if g not in ("market", "scenario", "strategy_id", "date", "guild_id"):
                raise ToolError("bad_group_by", f"不支持按 {g} 分组", "可选: market, scenario, strategy_id, date, guild_id")
        if granularity == "day" and "date" not in group_by:
            group_by = ["date"] + group_by
        spec = METRICS[key]
        if strategy_id and spec["level"] == "scenario":
            raise ToolError("bad_filter", f"{key} 是场景级指标，不能按 strategy_id 过滤", "改用 scenario 过滤")

        if guild_id or "guild_id" in group_by:
            if key in ("precision_est", "recall_est", "reports"):
                raise ToolError("bad_filter", f"{key} 不支持公会维度", "公会维度支持 sessions/hits/manual_reviews/enforcements/appeals/overturns 及其比率")
            rows = self._guild_rows(key, spec, start, end, markets, scenario, strategy_id, guild_id, group_by)
        elif key in ("precision_est", "recall_est"):
            rows = self._estimate_rows(key, start, end, markets, scenario, strategy_id, group_by)
        else:
            rows = self._count_rows(key, spec, start, end, markets, scenario, strategy_id, group_by)
        rows = rows[:400]
        return {"ref_id": self._ref("query_metrics"), "metric": key, "metric_zh": spec["zh"],
                "metric_version": spec["version"], "definition": spec["definition"],
                "filters": {"market": markets or "ALL", "scenario": scenario, "strategy_id": strategy_id, "guild_id": guild_id,
                            "start_date": start, "end_date": end, "window_clipped": clipped},
                "group_by": group_by, "rows": rows, "row_count": len(rows),
                "note": "low_sample=true 表示分母或样本量低于门槛，结论应降低置信度或升级。"}

    def _count_rows(self, key, spec, start, end, markets, scenario, strategy_id, group_by):
        if strategy_id or "strategy_id" in group_by:
            df = self.metrics_strategy
            if scenario:
                sids = [s["strategy_id"] for s in self.strategies if s["scenario"] == scenario]
                df = df[df.strategy_id.isin(sids)]
            if strategy_id:
                df = df[df.strategy_id == strategy_id]
            if key == "reports":
                raise ToolError("bad_filter", f"{key} 不按策略拆分", "去掉 strategy_id 过滤/分组，改用 scenario")
        else:
            df = self.metrics
            if scenario:
                df = df[df.scenario == scenario]
        df = df[(df.date >= start) & (df.date <= end)]
        if markets:
            df = df[df.market.isin(markets)]
        if df.empty:
            raise ToolError("no_data", "过滤条件下没有数据", "放宽市场/场景/日期过滤")
        keys = [g for g in group_by if g in df.columns] or ["_all"]
        if keys == ["_all"]:
            df = df.assign(_all="total")
        # sessions is market-level: de-duplicate per (date, market)
        sess = df.drop_duplicates(["date", "market"]).groupby(keys).sessions.sum() if "sessions" in df.columns else None
        if spec["kind"] == "count":
            if key == "sessions":
                agg = sess
            else:
                agg = df.groupby(keys)[key].sum()
            out = [dict(zip(keys, k if isinstance(k, tuple) else (k,)), value=int(v), low_sample=False)
                   for k, v in agg.items()]
        else:
            num = sess if spec["num"] == "sessions" else df.groupby(keys)[spec["num"]].sum()
            den = sess if spec["den"] == "sessions" else df.groupby(keys)[spec["den"]].sum()
            out = []
            for k in num.index:
                n, d = int(num[k]), int(den[k])
                out.append(dict(zip(keys, k if isinstance(k, tuple) else (k,)),
                                value=round(n / d, 4) if d else None, numerator=n, denominator=d,
                                low_sample=d < (20 if spec["den"] != "sessions" else 150)))
        for r in out:
            r.pop("_all", None)
        return out

    def _guild_rows(self, key, spec, start, end, markets, scenario, strategy_id, guild_id, group_by):
        """Counts computed from raw tables joined with the session's guild (metrics tables have no guild dimension)."""
        def base(df, scn_col=None, strat_col=None):
            df = df[(df.date >= start) & (df.date <= end)]
            if markets:
                df = df[df.market.isin(markets)]
            if scenario and scn_col:
                df = df[df[scn_col] == scenario]
            if strategy_id and strat_col:
                df = df[df[strat_col] == strategy_id]
            df = self._with_guild(df)
            if guild_id:
                df = df[df.guild_id == guild_id]
            return df
        sess = base(self.sessions)
        tables = {
            "sessions": sess,
            "hits": base(self.hits, "scenario", "strategy_id"),
            "manual_reviews": base(self.hits[self.hits.action == "manual_review"], "scenario", "strategy_id"),
            "enforcements": base(self.enforcements, "scenario", "primary_strategy_id"),
            "appeals": base(self.appeals, "scenario", "strategy_id"),
            "overturns": base(self.appeals[self.appeals.result == "overturned"], "scenario", "strategy_id"),
        }
        def count(name, keys):
            df = tables[name]
            ks = [k for k in keys if k in df.columns or k == "strategy_id"]
            if name == "enforcements" and "strategy_id" in ks:
                df = df.rename(columns={"primary_strategy_id": "strategy_id"})
            ks = [k for k in ks if k in df.columns]
            if not ks:
                return {(): len(df)}, []
            g = df.groupby(ks).size()
            return {(k if isinstance(k, tuple) else (k,)): int(v) for k, v in g.items()}, ks
        keys = group_by or []
        if spec["kind"] == "count":
            c, ks = count(key, keys)
            out = [dict(zip(ks, k), value=v, low_sample=False) for k, v in c.items()]
        else:
            num, ks = count(spec["num"], keys)
            den_keys = [k for k in keys if k in ("date", "market", "guild_id")] if spec["den"] == "sessions" else keys
            den, dks = count(spec["den"], den_keys)
            out = []
            for k, n in num.items():
                dk = tuple(v for kk, v in zip(ks, k) if kk in dks)
                d = den.get(dk, 0)
                out.append(dict(zip(ks, k), value=round(n / d, 4) if d else None, numerator=n, denominator=d,
                                low_sample=d < (20 if spec["den"] != "sessions" else 150)))
        for r in out:
            if r.get("guild_id") == "":
                r["guild_id"] = "(无公会)"
            elif r.get("guild_id"):
                r["guild_name"] = self.guild_by_id.get(r["guild_id"], {}).get("name")
        return sorted(out, key=lambda r: (r.get("date", ""), -(r["value"] or 0)))

    def _estimate_rows(self, key, start, end, markets, scenario, strategy_id, group_by):
        qa = self.qa[(self.qa.date >= start) & (self.qa.date <= end)]
        if markets:
            qa = qa[qa.market.isin(markets)]
        keys = [g for g in group_by if g != "strategy_id" or key == "precision_est"]
        if key == "precision_est":
            h = qa[qa.stratum == "hit"]
            if scenario:
                h = h[h.scenario == scenario]
            if strategy_id:
                h = h[h.strategy_id == strategy_id]
            if h.empty:
                raise ToolError("no_data", "该过滤条件下没有 QA 抽检样本", "放宽窗口或市场")
            h = h.assign(tp=(h.qa_label == h.scenario).astype(int))
            g = h.groupby(keys) if keys else h.assign(_a=1).groupby("_a")
            out = []
            for k, sub in g:
                k = k if isinstance(k, tuple) else (k,)
                row = dict(zip(keys, k)) if keys else {}
                row.update(value=round(sub.tp.mean(), 3), n=int(len(sub)), low_sample=len(sub) < 30)
                out.append(row)
            return out
        # recall_est (scenario-level)
        if not scenario:
            raise ToolError("bad_filter", "recall_est 需要指定 scenario", "例如 scenario='gambling'")
        keys = [g for g in keys if g != "scenario"]
        hitq = qa[(qa.stratum == "hit") & (qa.scenario == scenario)]
        nonq = qa[qa.stratum.str.startswith("nonhit")]
        hits = self.hits[(self.hits.scenario == scenario) & (self.hits.date >= start) & (self.hits.date <= end)]
        if markets:
            hits = hits[hits.market.isin(markets)]
        hit_sessions = hits.drop_duplicates("session_id")
        out = []
        groups = [((), None)] if not keys else None
        if keys:
            frames = {k if isinstance(k, tuple) else (k,): None for k, _ in hit_sessions.groupby(keys)}
            for k, _ in nonq.groupby(keys):
                frames.setdefault(k if isinstance(k, tuple) else (k,), None)
            groups = [(k, None) for k in sorted(frames)]
        for k, _ in groups:
            def filt(df):
                for kk, vv in zip(keys, k):
                    df = df[df[kk] == vv]
                return df
            hs, hq, nq = filt(hit_sessions), filt(hitq), filt(nonq)
            prec = (hq.qa_label == scenario).mean() if len(hq) else np.nan
            tp_est = len(hs) * (prec if not np.isnan(prec) else 0)
            missed = float(nq[nq.qa_label == scenario].weight.sum())
            val = tp_est / (tp_est + missed) if (tp_est + missed) > 0 else None
            row = dict(zip(keys, k))
            row.update(value=round(val, 3) if val is not None else None, hit_sessions=int(len(hs)),
                       precision_est=round(prec, 3) if not np.isnan(prec) else None, hit_qa_n=int(len(hq)),
                       nonhit_qa_n=int(len(nq)), missed_est=round(missed, 1),
                       low_sample=(len(hq) < 30 or len(nq) < 50))
            out.append(row)
        return out

    def get_strategy_config(self, strategy_id=None, scenario=None, market=None, status=None):
        """Read strategy conditions, thresholds, actions, versions and owners."""
        strategy_id = self._check_strategy(strategy_id)
        scenario = self._check_scenario(scenario)
        markets = self._check_market(market)
        out = []
        for s in self.strategies:
            if strategy_id and s["strategy_id"] != strategy_id:
                continue
            if scenario and s["scenario"] != scenario:
                continue
            if markets and s["markets"] != ["ALL"] and not set(markets) & set(s["markets"]):
                continue
            if status and s["status"] != status:
                continue
            rec = dict(s)
            rec["model"] = f"model_{s['scenario']}"
            rec["condition_text"] = " AND ".join(
                [f"{s['score_field']} >= {s['threshold']}"] + [f"{c['field']} {c['op']} {c['value']}" for c in s["extra_conditions"]])
            out.append(rec)
        if not out:
            raise ToolError("no_data", "没有匹配的策略", "检查 strategy_id/scenario/market")
        return {"ref_id": self._ref("get_strategy_config"), "count": len(out), "strategies": out}

    def get_hit_logs(self, start_date, end_date, strategy_id=None, market=None, scenario=None, limit=50,
                     overlap_analysis=False, guild_id=None):
        """Pull hit and disposition records, with per-action/strategy summary and optional overlap analysis."""
        start, end, _ = self._check_window(start_date, end_date)
        strategy_id = self._check_strategy(strategy_id)
        markets = self._check_market(market)
        scenario = self._check_scenario(scenario)
        h = self.hits[(self.hits.date >= start) & (self.hits.date <= end)]
        if strategy_id:
            h = h[h.strategy_id == strategy_id]
        if markets:
            h = h[h.market.isin(markets)]
        if scenario:
            h = h[h.scenario == scenario]
        guild_id = self._check_guild(guild_id)
        h = self._with_guild(h)
        if guild_id:
            h = h[h.guild_id == guild_id]
        limit = int(min(max(limit, 1), 200))
        res = {"ref_id": self._ref("get_hit_logs"), "filters": {"strategy_id": strategy_id, "market": markets or "ALL",
               "scenario": scenario, "guild_id": guild_id, "start_date": start, "end_date": end},
               "summary": {"hits": int(len(h)), "distinct_sessions": int(h.session_id.nunique()),
                           "by_action": h.action.value_counts().to_dict(),
                           "by_strategy": h.strategy_id.value_counts().head(20).to_dict(),
                           "by_market": h.market.value_counts().to_dict(),
                           "by_guild": h.guild_id.replace("", "(无公会)").value_counts().head(10).to_dict(),
                           "by_model_version": h.model_version.value_counts().to_dict()},
               "rows": h.sort_values("date", ascending=False).head(limit).to_dict("records"),
               "truncated": len(h) > limit}
        if overlap_analysis:
            if not strategy_id:
                raise ToolError("bad_filter", "overlap_analysis 需要 strategy_id")
            mine = set(h.session_id)
            others = self.hits[(self.hits.date >= start) & (self.hits.date <= end) & self.hits.session_id.isin(mine)
                               & (self.hits.strategy_id != strategy_id)]
            ov = []
            for sid, g in others.groupby("strategy_id"):
                co = g.session_id.nunique()
                tot_other = self.hits[(self.hits.date >= start) & (self.hits.date <= end) & (self.hits.strategy_id == sid)].session_id.nunique()
                ov.append({"strategy_id": sid, "scenario": self.by_id[sid]["scenario"], "co_hit_sessions": int(co),
                           "share_of_this": round(co / max(len(mine), 1), 3), "share_of_other": round(co / max(tot_other, 1), 3),
                           "same_action": self.by_id[sid]["action"] == self.by_id[strategy_id]["action"]})
            res["overlap"] = sorted(ov, key=lambda r: -r["co_hit_sessions"])[:10]
            res["overlap_note"] = "share_of_this = 本策略命中 session 中也被对方命中的比例；两者都高且动作相同 => 重复处罚风险"
        return res

    def sample_cases(self, start_date, end_date, strategy_id=None, market=None, scenario=None, n=15,
                     appealed=None, overturned=None, action=None, host_tier=None, guild_id=None):
        """Sample typical cases (sessions) with scores, thresholds, actions, appeal results and QA labels."""
        start, end, _ = self._check_window(start_date, end_date)
        strategy_id = self._check_strategy(strategy_id)
        markets = self._check_market(market)
        scenario = self._check_scenario(scenario)
        h = self.hits[(self.hits.date >= start) & (self.hits.date <= end)]
        if strategy_id:
            h = h[h.strategy_id == strategy_id]
        if markets:
            h = h[h.market.isin(markets)]
        if scenario:
            h = h[h.scenario == scenario]
        if action:
            h = h[h.action == action]
        ap = self.appeals.set_index("session_id")
        if appealed is not None:
            h = h[h.session_id.isin(ap.index) == bool(appealed)]
        if overturned is not None:
            ov = set(ap[ap.result == "overturned"].index)
            h = h[h.session_id.isin(ov) == bool(overturned)]
        if host_tier:
            h = h[h.session_id.map(self.sess_idx.host_tier) == host_tier]
        guild_id = self._check_guild(guild_id)
        if guild_id:
            h = h[h.session_id.map(self.sess_idx.guild_id) == guild_id]
        if h.empty:
            raise ToolError("no_data", "没有符合条件的 case", "放宽过滤条件")
        n = int(min(max(n, 1), 30))
        picked = h.drop_duplicates("session_id").sample(n=min(n, h.session_id.nunique()), random_state=7)
        qa_lab = self.qa.drop_duplicates("session_id").set_index("session_id").qa_label
        cases = []
        for r in picked.itertuples():
            s = self.sess_idx.loc[r.session_id]
            a = ap.loc[r.session_id] if r.session_id in ap.index else None
            cases.append({"session_id": r.session_id, "hit_id": r.hit_id, "date": r.date, "market": r.market,
                          "host_tier": s.host_tier, "guild_id": s.guild_id or None, "strategy_id": r.strategy_id, "strategy_version": r.strategy_version,
                          "score": float(r.score), "threshold": float(r.threshold), "model_version": r.model_version,
                          "action": r.action, "appeal": None if a is None else {"appeal_id": a.appeal_id, "result": a.result},
                          "qa_label": qa_lab.get(r.session_id, None),
                          "features": {k: (int(s[k]) if float(s[k]).is_integer() else float(s[k])) for k in
                                       ("account_age_days", "viewers_peak", "gift_amount", "report_count",
                                        "external_link_flag", "new_device_flag", "keyword_hits")}})
        gshare = h.drop_duplicates("session_id").session_id.map(self.sess_idx.guild_id).replace("", "(无公会)").value_counts(normalize=True)
        return {"ref_id": self._ref("sample_cases"), "population": int(h.session_id.nunique()), "n": len(cases),
                "contains_head_host": any(c["host_tier"] == "head" for c in cases),
                "guild_share_in_population": {k: round(float(v), 3) for k, v in gshare.head(5).items()},
                "score_summary": {"mean": round(float(h.score.mean()), 3), "p50": round(float(h.score.median()), 3),
                                  "min": round(float(h.score.min()), 3)},
                "cases": cases}

    def get_change_log(self, start_date, end_date, scenario=None, market=None, strategy_id=None, change_type=None):
        """Strategy launches/updates, model upgrades, policy updates and feature changes in a window."""
        try:
            s, e = _d(start_date), _d(end_date)
        except Exception:
            raise ToolError("bad_date", "日期格式错误", "YYYY-MM-DD")
        scenario = self._check_scenario(scenario)
        markets = self._check_market(market)
        strategy_id = self._check_strategy(strategy_id)
        out = []
        for c in self.change_log:
            if not (s <= _d(c["date"]) <= e):
                continue
            if scenario and c["scenario"] != scenario:
                continue
            if markets and c["markets"] != ["ALL"] and not set(markets) & set(c["markets"]):
                continue
            if strategy_id and c["strategy_id"] != strategy_id:
                continue
            if change_type and c["type"] != change_type:
                continue
            out.append(c)
        return {"ref_id": self._ref("get_change_log"), "count": len(out), "changes": out,
                "types": ["strategy_launch", "strategy_update", "model_update", "policy_update", "feature_pipeline"]}

    def score_drift(self, scenario, baseline_start, baseline_end, current_start, current_end, market=None):
        """Compare the model-score distribution between two windows (PSI, quantiles, share above thresholds)."""
        scenario = self._check_scenario(scenario)
        markets = self._check_market(market)
        b0, b1, _ = self._check_window(baseline_start, baseline_end)
        c0, c1, _ = self._check_window(current_start, current_end)
        col = f"score_{scenario}"
        df = self.sessions
        if markets:
            df = df[df.market.isin(markets)]
        base = df[(df.date >= b0) & (df.date <= b1)]
        cur = df[(df.date >= c0) & (df.date <= c1)]
        if len(base) < 50 or len(cur) < 50:
            raise ToolError("insufficient_sample", f"样本不足：baseline={len(base)}, current={len(cur)}", "扩大窗口或市场")
        def ver(d0, d1):
            vs = [m for m in self.meta["model_versions"] if m["scenario"] == scenario]
            active = sorted({v["version"] for v in vs if v["effective_from"] <= d1 and
                             not any(w["effective_from"] > v["effective_from"] and w["effective_from"] <= d0 for w in vs)})
            return active
        def stats(x):
            x = x.values
            return {"n": int(len(x)), "mean": round(float(x.mean()), 4), "p50": round(float(np.quantile(x, .5)), 4),
                    "p90": round(float(np.quantile(x, .9)), 4), "p99": round(float(np.quantile(x, .99)), 4)}
        thr = []
        for st in self.strategies:
            if st["scenario"] == scenario and st["status"] == "active":
                thr.append({"strategy_id": st["strategy_id"], "threshold": st["threshold"], "action": st["action"],
                            "share_above_baseline": round(float((base[col] >= st["threshold"]).mean()), 4),
                            "share_above_current": round(float((cur[col] >= st["threshold"]).mean()), 4)})
        for t in thr:
            t["ratio"] = round(t["share_above_current"] / t["share_above_baseline"], 2) if t["share_above_baseline"] else None
        rep_ids = set(self.reports[self.reports.reason == scenario].session_id)
        rb, rc = base[base.session_id.isin(rep_ids)], cur[cur.session_id.isin(rep_ids)]
        psi = _psi(base[col].values, cur[col].values)
        return {"ref_id": self._ref("score_drift"), "model": f"model_{scenario}", "market": markets or "ALL",
                "baseline": {"window": [b0, b1], "model_versions": ver(b0, b1), **stats(base[col])},
                "current": {"window": [c0, c1], "model_versions": ver(c0, c1), **stats(cur[col])},
                "psi": round(psi, 4), "drift_flag": psi > 0.1,
                "psi_note": "PSI < 0.1 稳定；0.1~0.25 轻微漂移；> 0.25 显著漂移",
                "share_above_thresholds": thr,
                "reported_population": {"note": "被举报为该场景的 session 的分数（召回侧信号）",
                                        "baseline": stats(rb[col]) if len(rb) >= 10 else {"n": int(len(rb)), "low_sample": True},
                                        "current": stats(rc[col]) if len(rc) >= 10 else {"n": int(len(rc)), "low_sample": True}}}

    def run_backtest(self, strategy_id, start_date, end_date, threshold=None, market=None):
        """Replay a strategy with an alternative threshold over historical sessions; estimate precision/recall on QA-labeled sessions."""
        strategy_id = self._check_strategy(strategy_id)
        start, end, _ = self._check_window(start_date, end_date)
        markets = self._check_market(market)
        st = self.by_id[strategy_id]
        new_thr = float(threshold) if threshold is not None else st["threshold"]
        if not (0 < new_thr < 1):
            raise ToolError("bad_threshold", "threshold 必须在 (0,1)")
        df = self.sessions[(self.sessions.date >= start) & (self.sessions.date <= end)]
        if st["markets"] != ["ALL"]:
            df = df[df.market.isin(st["markets"])]
        if markets:
            df = df[df.market.isin(markets)]
        mask = np.ones(len(df), dtype=bool)
        ops = {">=": np.greater_equal, ">": np.greater, "<=": np.less_equal, "<": np.less, "==": np.equal, "!=": np.not_equal}
        for c in st["extra_conditions"]:
            mask &= ops[c["op"]](df[c["field"]].values, c["value"])
        sc = df[st["score_field"]].values
        before, after = mask & (sc >= st["threshold"]), mask & (sc >= new_thr)
        lab = self.qa.drop_duplicates("session_id").set_index("session_id").qa_label
        labeled = df.session_id.map(lab)
        has = labeled.notna().values
        pos = (labeled.values == st["scenario"])
        def pr(sel):
            n = int((sel & has).sum())
            tp = int((sel & has & pos).sum())
            p = round(tp / n, 3) if n else None
            r = round(tp / max(int((has & pos).sum()), 1), 3)
            return {"labeled_hits": n, "precision_est": p, "recall_on_labeled": r}
        ap = self.appeals[self.appeals.strategy_id == strategy_id]
        ov = set(ap[ap.result == "overturned"].session_id)
        removed = df.session_id[before & ~after]
        days = max((_d(end) - _d(start)).days + 1, 1)
        return {"ref_id": self._ref("run_backtest"), "strategy_id": strategy_id, "window": [start, end],
                "market": markets or st["markets"], "threshold_before": st["threshold"], "threshold_after": new_thr,
                "hits_before": int(before.sum()), "hits_after": int(after.sum()),
                "hits_per_day_before": round(before.sum() / days, 1), "hits_per_day_after": round(after.sum() / days, 1),
                "delta_pct": round((after.sum() - before.sum()) / max(before.sum(), 1) * 100, 1),
                "before": pr(before), "after": pr(after), "labeled_positives": int((has & pos).sum()),
                "overturned_appeals_avoided": int(removed.isin(ov).sum()),
                "caveat": "precision/recall 只在 QA 已标注 session 子集上估算，样本有偏；仅供方向判断"}

    def reconstruct_case(self, session_id):
        """Rebuild one case: features, scores, model versions, strategies hit, enforcement chain, appeal, QA label."""
        if session_id not in self.known["session_id"]:
            raise ToolError("unknown_session", f"session {session_id} 不存在")
        s = self.sess_idx.loc[session_id]
        h = self.hits[self.hits.session_id == session_id]
        e = self.enforcements[self.enforcements.session_id == session_id]
        a = self.appeals[self.appeals.session_id == session_id]
        q = self.qa[self.qa.session_id == session_id]
        d = s.date
        scores = {}
        for sc in self.meta["scenarios"]:
            vs = [m["version"] for m in self.meta["model_versions"] if m["scenario"] == sc and m["effective_from"] <= d]
            scores[sc] = {"score": float(s[f"score_{sc}"]), "model": f"model_{sc}", "model_version": vs[-1] if vs else None}
        chain = [{"step": "strategy_hit", **r} for r in h.drop(columns=["session_id"]).to_dict("records")]
        if len(e):
            chain.append({"step": "enforcement", **e.iloc[0].drop("session_id").to_dict()})
        if len(a):
            chain.append({"step": "appeal", **a.iloc[0].drop("session_id").to_dict()})
        changes = [c for c in self.change_log if (_d(d) - timedelta(days=7)) <= _d(c["date"]) <= _d(d)
                   and (c["scenario"] in set(h.scenario) or c["strategy_id"] in set(h.strategy_id))]
        flags = []
        if s.host_tier == "head":
            flags.append("head_host: 头部主播，需人工确认")
        if len(a) and a.iloc[0].result == "overturned":
            flags.append("appeal_overturned: 申诉已推翻，疑似误杀")
        for r in h.itertuples():
            if r.score - r.threshold < 0.05:
                flags.append(f"{r.strategy_id} 分数 {r.score:.3f} 仅略高于阈值 {r.threshold}（边界 case）")
        return {"ref_id": self._ref("reconstruct_case"), "session_id": session_id,
                "session": {"date": d, "market": s.market, "host_id": s.host_id, "host_tier": s.host_tier,
                            "guild_id": s.guild_id or None, "guild": self.guild_by_id.get(s.guild_id), "duration_min": int(s.duration_min),
                            "features": {k: (int(s[k]) if float(s[k]).is_integer() else float(s[k])) for k in
                                         ("account_age_days", "viewers_peak", "gift_amount", "report_count",
                                          "external_link_flag", "new_device_flag", "keyword_hits", "legacy_brand_list_match")}},
                "scores": scores, "chain": chain, "related_changes": changes,
                "qa_label": q.iloc[0].qa_label if len(q) else None, "escalation_flags": flags,
                "platform_link": f"platform://case/{session_id}"}

    def get_guild_info(self, guild_id=None, market=None):
        """Guild registry: id, market, name, size tier, 28-day sessions, head hosts, manager."""
        guild_id = self._check_guild(guild_id)
        markets = self._check_market(market)
        out = [g for g in self.guilds if (not guild_id or g["guild_id"] == guild_id) and (not markets or g["market"] in markets)]
        if not out:
            raise ToolError("no_data", "没有匹配的公会")
        return {"ref_id": self._ref("get_guild_info"), "count": len(out), "guilds": sorted(out, key=lambda g: -g["sessions_28d"])[:50]}

    def draft_change(self, strategy_id, proposed_threshold, rationale, evidence_ids, proposed_action=None):
        """Create a change-request DRAFT (never applied). Risk tier decides who must approve."""
        strategy_id = self._check_strategy(strategy_id)
        st = self.by_id[strategy_id]
        if not evidence_ids:
            raise ToolError("missing_evidence", "变更草稿必须附证据 id")
        sev = self.meta["action_severity"]
        act = proposed_action or st["action"]
        if act not in sev:
            raise ToolError("bad_action", f"未知动作 {act}", f"可选: {list(sev)}")
        if sev[act] >= 4 and st["markets"] == ["ALL"]:
            risk, approvers = "high", ["仅输出建议：需策略负责人 + 运营 + 治理质量评审会"]
        elif sev[act] >= 2:
            risk, approvers = "medium", ["策略负责人", "运营负责人"]
        else:
            risk, approvers = "low", ["策略负责人"]
        os.makedirs(DRAFT_DIR, exist_ok=True)
        n = len([f for f in os.listdir(DRAFT_DIR) if f.endswith(".json")]) + 1
        draft = {"draft_id": f"CHG-{n:04d}", "status": "draft_pending_approval", "created_by": self.user,
                 "strategy_id": strategy_id, "current": {"threshold": st["threshold"], "action": st["action"], "version": st["version"]},
                 "proposed": {"threshold": proposed_threshold, "action": act}, "rationale": rationale,
                 "evidence_ids": evidence_ids, "risk_level": risk, "required_approvals": approvers,
                 "note": "草稿不会生效；提交需走现有变更审批流。"}
        with open(os.path.join(DRAFT_DIR, draft["draft_id"] + ".json"), "w") as f:
            json.dump(draft, f, ensure_ascii=False, indent=1)
        draft["ref_id"] = self._ref("draft_change")
        return draft

    # ------------------------------------------------------------------ dispatch
    def call(self, name, args):
        fn = {t["name"]: getattr(self, t["name"]) for t in TOOL_SPECS}.get(name)
        if fn is None:
            return {"error": "unknown_tool", "message": f"工具 {name} 不存在"}
        try:
            return fn(**args)
        except ToolError as e:
            return e.to_dict()
        except TypeError as e:
            return {"error": "bad_arguments", "message": str(e)}


def _p(**props):
    return {"type": "object", "properties": props, "additionalProperties": False}


_DATE = {"type": "string", "description": "YYYY-MM-DD"}
_MKT = {"type": ["string", "null"], "description": "市场代码，如 US/BR/ID/GB/VN/KE；null 表示全部"}
_SCN = {"type": ["string", "null"], "description": "场景: adult_content, minor_safety, scam_fraud, violence, spam_bot, gambling, hate_harassment, counterfeit"}
_SID = {"type": ["string", "null"], "description": "策略编号，如 S-102"}
_GID = {"type": ["string", "null"], "description": "公会编号，如 G-BR-003；null 表示不按公会过滤"}

TOOL_SPECS = [
    {"name": "query_metrics", "description": "按天或汇总查询治理指标（sessions/hits/manual_reviews/enforcements/appeals/overturns/reports/"
     "hit_rate/enforcement_rate/appeal_rate/overturn_rate/overturn_per_enforcement/precision_est/recall_est）。可按 market/scenario/strategy_id/guild_id 分组。返回口径定义和 ref_id。"
     "中文别名可能对应多个口径，此时返回 ambiguous_metric 错误，需选定一个。",
     "input_schema": {**_p(metric={"type": "string"}, start_date=_DATE, end_date=_DATE, market=_MKT, scenario=_SCN,
                           strategy_id=_SID, guild_id=_GID,
                           group_by={"type": "array", "items": {"type": "string", "enum": ["market", "scenario", "strategy_id", "date", "guild_id"]}},
                           granularity={"type": "string", "enum": ["day", "total"]}),
                      "required": ["metric", "start_date", "end_date"]}},
    {"name": "get_strategy_config", "description": "读取策略条件、阈值、动作、版本历史、owner。",
     "input_schema": {**_p(strategy_id=_SID, scenario=_SCN, market=_MKT, status={"type": ["string", "null"], "enum": ["active", "paused", None]}),
                      "required": []}},
    {"name": "get_hit_logs", "description": "拉取策略命中与处置记录；overlap_analysis=true 时给出与其他策略的重叠分析（需 strategy_id）。",
     "input_schema": {**_p(start_date=_DATE, end_date=_DATE, strategy_id=_SID, market=_MKT, scenario=_SCN, guild_id=_GID,
                           limit={"type": "integer"}, overlap_analysis={"type": "boolean"}),
                      "required": ["start_date", "end_date"]}},
    {"name": "sample_cases", "description": "抽样典型 case（带分数、阈值、模型版本、处置、申诉结果、QA 标签、关键特征）。",
     "input_schema": {**_p(start_date=_DATE, end_date=_DATE, strategy_id=_SID, market=_MKT, scenario=_SCN, n={"type": "integer"},
                           appealed={"type": ["boolean", "null"]}, overturned={"type": ["boolean", "null"]},
                           action={"type": ["string", "null"]}, host_tier={"type": ["string", "null"], "enum": ["head", "mid", "tail", None]},
                           guild_id=_GID),
                      "required": ["start_date", "end_date"]}},
    {"name": "get_change_log", "description": "查询窗口内的策略上线/更新、模型升级、policy 更新、特征管线变更。",
     "input_schema": {**_p(start_date=_DATE, end_date=_DATE, scenario=_SCN, market=_MKT, strategy_id=_SID,
                           change_type={"type": ["string", "null"]}),
                      "required": ["start_date", "end_date"]}},
    {"name": "score_drift", "description": "比较两个窗口的模型分分布（PSI、分位数、超过各策略阈值的比例、被举报人群的分数）。",
     "input_schema": {**_p(scenario={"type": "string"}, baseline_start=_DATE, baseline_end=_DATE, current_start=_DATE,
                           current_end=_DATE, market=_MKT),
                      "required": ["scenario", "baseline_start", "baseline_end", "current_start", "current_end"]}},
    {"name": "run_backtest", "description": "用历史数据回放策略在新阈值下的命中量，并在 QA 已标注子集上估算 precision/recall 变化。",
     "input_schema": {**_p(strategy_id={"type": "string"}, start_date=_DATE, end_date=_DATE,
                           threshold={"type": ["number", "null"]}, market=_MKT),
                      "required": ["strategy_id", "start_date", "end_date"]}},
    {"name": "reconstruct_case", "description": "还原单个 case：特征、各模型分与版本、命中的策略、处罚链路、申诉、QA 标签、相关变更。",
     "input_schema": {**_p(session_id={"type": "string"}), "required": ["session_id"]}},
    {"name": "get_guild_info", "description": "查询公会信息（市场、规模、头部主播数、运营负责人）。",
     "input_schema": {**_p(guild_id=_GID, market=_MKT), "required": []}},
    {"name": "draft_change", "description": "生成策略变更单草稿（只生成草稿，不提交、不生效），按影响面给出风险等级和审批人。",
     "input_schema": {**_p(strategy_id={"type": "string"}, proposed_threshold={"type": "number"}, rationale={"type": "string"},
                           evidence_ids={"type": "array", "items": {"type": "string"}}, proposed_action={"type": ["string", "null"]}),
                      "required": ["strategy_id", "proposed_threshold", "rationale", "evidence_ids"]}},
]
