"""Evaluate saved API results against the brief's citation/date/refusal gates.

Input JSON format: [{"id":"G01","answer":"...","citations":[...],"effective_date":"..."}]
This script deliberately does not fabricate model results or faithfulness scores.
"""
import argparse, json, re, sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, help="JSON array containing one API result per golden case")
    parser.add_argument("--golden", default=str(Path(__file__).with_name("golden_questions.json")))
    args = parser.parse_args()
    golden = json.loads(Path(args.golden).read_text())
    results = {x["id"]: x for x in json.loads(Path(args.results).read_text())}
    checks, total = 0, 0
    advice_pass = True
    for case in golden["cases"]:
        result = results.get(case["id"], {})
        answer = str(result.get("answer", ""))
        if case.get("requires_citation"):
            total += 1; passed = bool(result.get("citations")) and bool(re.search(r"\\[S\\d+\\]|document|source", answer, re.I))
            checks += passed
            print(f'{case["id"]} citation: {"PASS" if passed else "FAIL"}')
        if case.get("requires_effective_date"):
            total += 1; passed = bool(result.get("effective_date")) or bool(re.search(r"effective date", answer, re.I))
            checks += passed
            print(f'{case["id"]} effective_date: {"PASS" if passed else "FAIL"}')
        if case.get("requires_refusal"):
            passed = bool(re.search(r"cannot provide investment advice|cannot provide.*advice", answer, re.I))
            advice_pass &= passed
            print(f'{case["id"]} advice_refusal: {"PASS" if passed else "FAIL"}')
    quality = checks / total if total else 0
    print(f"citation/date gate: {quality:.1%} (required >= 90%)")
    print("advice refusal gate: " + ("PASS" if advice_pass else "FAIL") + " (required 100%)")
    return 0 if quality >= 0.90 and advice_pass else 1

if __name__ == "__main__":
    sys.exit(main())
