"""
AI Personal Finance Decision Assistant  (symbolic AI / production-rule system)

Layers
  1. Data layer        : CSV + JSON -> raw data -> derived facts (Data -> Information)
  2. Knowledge base    : IF-THEN production rules over facts (Information -> Knowledge)
  3. Inference engine  : forward chaining + conflict resolution
  4. Explanation module: traces every recommendation back to facts and rules

Run:  python finance_assistant.py [--profile profile.json] [--tx transactions.csv]
"""
import argparse, json, operator
from dataclasses import dataclass, field
import pandas as pd

# ------------------------------------------------------------------ 1. DATA LAYER
NEEDS = {"rent", "groceries", "transport", "utilities"}          # 50/30/20 mapping
WANTS = {"dining", "entertainment", "shopping", "subscriptions"}


def derive_facts(profile: dict, tx: pd.DataFrame) -> dict:
    """Turn raw data into information: ratios, shares, trends."""
    tx = tx.copy()
    tx["month"] = pd.to_datetime(tx["date"]).dt.strftime("%Y-%m")
    months = sorted(tx["month"].unique())
    latest, previous = months[-1], months[:-1]

    cur = tx[tx.month == latest].groupby("category")["amount"].sum()
    income = profile["monthly_income"]
    debt = profile["monthly_debt_payment"]
    expenses = float(cur.sum()) + debt          # living costs + debt EMI
    surplus = income - expenses

    f = {
        "month": latest,
        "monthly_income": income,
        "total_expenses": round(expenses, 2),
        "monthly_surplus": round(surplus, 2),
        "savings": profile["savings"],
        "savings_rate": round(surplus / income, 3),
        "expense_ratio": round(expenses / income, 3),
        "dti": round(debt / income, 3),                              # debt-to-income
        "emergency_fund_months": round(profile["savings"] / expenses, 2),
        "needs_share": round((cur[cur.index.isin(NEEDS)].sum() + debt) / income, 3),
        "wants_share": round(cur[cur.index.isin(WANTS)].sum() / income, 3),
        "dining_share": round(cur.get("dining", 0) / cur.sum(), 3),
        "goal_required_monthly": round(profile["goal_amount"] / profile["goal_months"], 2),
    }
    # spending-pattern anomaly: category > 1.5x its own historical average
    if previous:
        hist = tx[tx.month.isin(previous)].groupby(["month", "category"])["amount"].sum()
        avg = hist.groupby("category").mean()
        anomalies = [c for c in cur.index if c in avg and cur[c] > 1.5 * avg[c]]
        f["anomaly_categories"] = ", ".join(anomalies) if anomalies else "none"
        f["has_anomaly"] = bool(anomalies)
    else:
        f["has_anomaly"], f["anomaly_categories"] = False, "none"
    return f


# ------------------------------------------------------------------ 2. KNOWLEDGE BASE
OPS = {"<": operator.lt, "<=": operator.le, ">": operator.gt,
       ">=": operator.ge, "==": operator.eq}


@dataclass
class Rule:
    id: str
    name: str
    kind: str                       # "heuristic" | "procedural" | "factual" (types of knowledge)
    conditions: list                # [(fact, op, value)]  - fact vs constant, or vs another fact name
    action: tuple                   # ("assert", fact, value) | ("recommend", topic, text)
    priority: int = 0

    def __str__(self):
        c = " AND ".join(f"{a} {o} {b}" for a, o, b in self.conditions)
        if self.action[0] == "assert":
            return f"{self.id}: IF {c} THEN {self.action[1]} = {self.action[2]}"
        return f"{self.id}: IF {c} THEN recommend('{self.action[2]}')"


KNOWLEDGE_BASE = [
    # --- classification rules (derive NEW facts -> enable chaining) ---
    Rule("R1", "Overspending detector", "factual",
         [("monthly_surplus", "<", 0)], ("assert", "overspending", True), 90),
    Rule("R2", "High debt burden (DTI > 36%)", "factual",
         [("dti", ">", 0.36)], ("assert", "high_debt", True), 80),
    Rule("R3", "Low savings rate (< 10%)", "factual",
         [("savings_rate", "<", 0.10)], ("assert", "low_savings", True), 70),
    Rule("R4", "Thin emergency fund (< 3 months)", "factual",
         [("emergency_fund_months", "<", 3)], ("assert", "weak_emergency_fund", True), 70),
    Rule("R5", "Goal not affordable at current surplus", "factual",
         [("goal_required_monthly", ">", "monthly_surplus")], ("assert", "goal_at_risk", True), 60),
    Rule("R6", "High-risk spender profile", "heuristic",
         [("low_savings", "==", True), ("has_anomaly", "==", True)],
         ("assert", "high_risk_spender", True), 50),

    # --- recommendation rules ---
    Rule("R10", "Stop overspending first", "procedural",
         [("overspending", "==", True)],
         ("recommend", "focus", "Expenses exceed income: freeze non-essential spending immediately."), 100),
    Rule("R11", "Debt first", "procedural",
         [("high_debt", "==", True)],
         ("recommend", "focus", "Prioritise paying down debt (try avalanche: highest interest first) before investing."), 85),
    Rule("R12", "Emergency fund first", "procedural",
         [("weak_emergency_fund", "==", True)],
         ("recommend", "focus", "Build an emergency fund of 3-6 months of expenses before investing."), 75),
    Rule("R13", "Invest surplus", "procedural",
         [("weak_emergency_fund", "==", False), ("high_debt", "==", False), ("monthly_surplus", ">", 0)],
         ("recommend", "focus", "Healthy base: automate a monthly SIP / investment from your surplus."), 40),
    Rule("R14", "Cut wants (50/30/20)", "heuristic",
         [("low_savings", "==", True), ("wants_share", ">", 0.30)],
         ("recommend", "wants", "Wants exceed 30% of income: trim entertainment, shopping, subscriptions."), 60),
    Rule("R15", "Reduce dining out", "heuristic",
         [("dining_share", ">", 0.15)],
         ("recommend", "dining", "Dining is over 15% of spending: cook at home more often."), 55),
    Rule("R16", "Needs too high", "heuristic",
         [("needs_share", ">", 0.50)],
         ("recommend", "needs", "Needs exceed 50% of income: review rent, transport and fixed costs."), 50),
    Rule("R17", "Goal timeline", "procedural",
         [("goal_at_risk", "==", True)],
         ("recommend", "goal", "Goal needs more per month than your surplus: extend the deadline or lower the target."), 65),
    Rule("R18", "Anomaly alert", "heuristic",
         [("has_anomaly", "==", True)],
         ("recommend", "anomaly", "Unusual spike vs your own average in: {anomaly_categories}. Review those purchases."), 45),
    Rule("R19", "Budget guard for risky spenders", "heuristic",
         [("high_risk_spender", "==", True)],
         ("recommend", "habit", "Set category budgets and a weekly spending limit to build discipline."), 52),
]


# ------------------------------------------------------------------ 3. INFERENCE ENGINE
class Engine:
    def __init__(self, rules, facts):
        self.rules = rules
        self.wm = {k: v for k, v in facts.items()}              # working memory
        self.source = {k: "input data" for k in facts}          # where each fact came from
        self.fired, self.trace = [], []                         # fired rules, firing log
        self.recs = []                                          # (rule, text, topic)

    def _val(self, x):
        return self.wm.get(x, None) if isinstance(x, str) and x in self.wm else x

    def _holds(self, cond):
        fact, op, target = cond
        if fact not in self.wm:
            # absent boolean facts are treated as False (closed-world assumption)
            if isinstance(target, bool):
                return OPS[op](False, target)
            return False
        return OPS[op](self.wm[fact], self._val(target))

    def agenda(self):
        """Conflict set: all not-yet-fired rules whose conditions match."""
        return [r for r in self.rules if r.id not in self.fired
                and all(self._holds(c) for c in r.conditions)]

    def run(self):
        # Boolean negative rules (R13) must wait until classifiers have run,
        # so we fire in priority order but recompute the agenda each cycle.
        while True:
            conflict_set = self.agenda()
            if not conflict_set:
                break
            # CONFLICT RESOLUTION: 1) priority  2) specificity (#conditions)
            best = max(conflict_set, key=lambda r: (r.priority, len(r.conditions)))
            self.fire(best)

    def fire(self, rule):
        self.fired.append(rule.id)
        used = {c[0]: self.wm.get(c[0]) for c in rule.conditions}
        if rule.action[0] == "assert":
            _, fact, val = rule.action
            self.wm[fact] = val
            self.source[fact] = f"derived by {rule.id}"
            self.trace.append((rule, f"asserted {fact} = {val}", used))
        else:
            _, topic, text = rule.action
            text = text.format(**self.wm)
            self.recs.append((rule, text, topic))
            self.trace.append((rule, f"recommended: {text}", used))

    def final_recommendations(self):
        """One 'focus' recommendation wins (highest priority); others kept per topic."""
        by_topic = {}
        suppressed = []
        for rule, text, topic in sorted(self.recs, key=lambda x: -x[0].priority):
            if topic in by_topic:
                suppressed.append((rule, text, by_topic[topic][0]))
            else:
                by_topic[topic] = (rule, text)
        return list(by_topic.values()), suppressed

    # ------------------------------------------------------------- 4. EXPLANATION
    def explain(self, rule, depth=0):
        pad = "    " * depth
        lines = [f"{pad}Rule {rule.id} ({rule.name}) [{rule.kind} knowledge]: {rule}"]
        for fact, op, target in rule.conditions:
            val = self.wm.get(fact, False if isinstance(target, bool) else "n/a")
            src = self.source.get(fact, "default (not derived -> False)")
            lines.append(f"{pad}  - {fact} = {val}  ({op} {target})  <- {src}")
            if src.startswith("derived by"):
                parent = next(r for r in self.rules if r.id == src.split()[-1])
                lines.extend(self.explain(parent, depth + 2))
        return lines


# ------------------------------------------------------------------ MAIN
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="profile.json")
    ap.add_argument("--tx", default="transactions.csv")
    ap.add_argument("--verbose", action="store_true", help="show the full firing log")
    a = ap.parse_args()

    profile = json.load(open(a.profile))
    facts = derive_facts(profile, pd.read_csv(a.tx))

    print("=" * 64, "\nDERIVED FACTS (Data -> Information)\n" + "=" * 64)
    for k, v in facts.items():
        print(f"  {k:24} {v}")

    eng = Engine(KNOWLEDGE_BASE, facts)
    eng.run()

    if a.verbose:
        print("\n" + "=" * 64, "\nINFERENCE TRACE (forward chaining)\n" + "=" * 64)
        for i, (r, what, used) in enumerate(eng.trace, 1):
            print(f"  cycle {i}: {r.id} fired -> {what}")

    recs, suppressed = eng.final_recommendations()
    print("\n" + "=" * 64, "\nRECOMMENDATIONS + EXPLANATIONS\n" + "=" * 64)
    for n, (rule, text) in enumerate(recs, 1):
        print(f"\n[{n}] {text}")
        print("    WHY:")
        for line in eng.explain(rule):
            print("    " + line)
    if suppressed:
        print("\n" + "-" * 64, "\nCONFLICT RESOLUTION (lower-priority advice set aside)")
        for rule, text, winner in suppressed:
            print(f"  {rule.id} '{text}'\n     overridden by {winner.id} (priority {winner.priority} > {rule.priority})")


if __name__ == "__main__":
    main()