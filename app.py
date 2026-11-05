"""Web interface (Flask) for the symbolic Finance Decision Assistant.
Run:  python app.py   ->  open http://127.0.0.1:5000
The reasoning is done ONLY by finance_assistant.py (rules + forward chaining)."""
import io
from flask import Flask, jsonify, render_template, request
import pandas as pd
import json
from finance_assistant import KNOWLEDGE_BASE, Engine, derive_facts

app = Flask(__name__)


def rule_json(r):
    return {"id": r.id, "name": r.name, "kind": r.kind, "priority": r.priority,
            "text": str(r), "type": r.action[0]}


def tree(eng, rule):
    """Explanation as a nested structure: rule -> conditions -> (derived-by rule)..."""
    node = {**rule_json(rule), "conditions": []}
    for fact, op, target in rule.conditions:
        val = eng.wm.get(fact, False if isinstance(target, bool) else None)
        src = eng.source.get(fact, "default (never derived, so False)")
        parent = None
        if src.startswith("derived by"):
            parent = tree(eng, next(r for r in eng.rules if r.id == src.split()[-1]))
        node["conditions"].append({"fact": fact, "op": op, "target": target,
                                   "value": val, "source": src, "parent": parent})
    return node


def build_transactions(categories):
    rows = []
    for c in categories:
        for m in ("2026-08", "2026-09"):
            rows.append((f"{m}-15", c["name"], float(c["usual"])))
        rows.append(("2026-10-15", c["name"], float(c["current"])))
    return pd.DataFrame(rows, columns=["date", "category", "amount"])


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/rules")
def rules():
    return jsonify([rule_json(r) for r in KNOWLEDGE_BASE])


@app.post("/api/analyze")
def analyze():
    payload = json.loads(request.form["payload"])
    profile = payload["profile"]
    if "csv" in request.files and request.files["csv"].filename:
        tx = pd.read_csv(io.BytesIO(request.files["csv"].read()))
    else:
        tx = build_transactions(payload["categories"])
    facts = derive_facts(profile, tx)
    eng = Engine(KNOWLEDGE_BASE, facts)
    eng.run()
    recs, suppressed = eng.final_recommendations()
    return jsonify({
        "facts": facts,
        "flags": {k: v for k, v in eng.wm.items() if isinstance(v, bool) and k not in facts},
        "recommendations": [{"text": t, "topic": r.action[1], "rule": rule_json(r),
                             "tree": tree(eng, r)} for r, t in recs],
        "suppressed": [{"text": t, "rule": rule_json(r), "by": rule_json(w)} for r, t, w in suppressed],
        "trace": [{"cycle": i, "rule": rule_json(r), "what": what}
                  for i, (r, what, _) in enumerate(eng.trace, 1)],
        "fired": eng.fired,
    })


if __name__ == "__main__":
    app.run(debug=True)

