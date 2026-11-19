# Evaluation evidence register

The source brief requires the following evidence. No results have been fabricated.

| Evidence | File / command | State |
|---|---|---|
| Golden questions | `evaluation/golden_questions.json` | Seeded with the five questions in the brief |
| Quality gate | `.github/workflows/quality-gate.yml` | Added; requires real `evaluation/results.json` |
| Citation/effective-date/advice gate | `python evaluation/evaluate.py --results evaluation/results.json` | Implemented; not run without real API output |
| Faithfulness report | `reports/faithfulness_report.md` | To be completed from automated judge output |
| Precision/recall experiment | `reports/retrieval_experiment.md` | To be completed with one-variable reranker comparison |
| Latency and guardrail comparison | `reports/guardrail_measurements.md` | To be measured on workshop hardware |
| Cost at 1,000 questions/day | `reports/cost_note.md` | Fill with actual provider pricing and token counts |
| Injection attacks | `reports/injection_attack_log.md` | Execute 3 poisoned-document cases |
| Responsible AI checklist | `reports/RESPONSIBLE_AI_CHECKLIST.md` | Checklist created; evidence/sign-off pending |
