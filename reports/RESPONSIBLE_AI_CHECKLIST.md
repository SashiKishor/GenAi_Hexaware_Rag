# Responsible AI & Security Checklist — Northbridge Bank

Status legend: **Implemented (code path present)** / **Pending evidence** / **Not yet demonstrated**.
Do not mark a control complete until its evidence is attached.

| NIST AI RMF function | Control / case-study requirement | Current implementation | Evidence required / status |
|---|---|---|---|
| Govern | Named owner and escalation channel for investment, tax and credit advice | Advice refusal messages in `rag/chain.py` and `security/input_guard.py` | Confirm human channels with bank owner; pending |
| Map | Banking scope, affected users, synthetic-only corpus | Prompt and domain classifier constrain to products, fees, KYC/AML | Data inventory and stakeholder sign-off; pending |
| Map | PII handling for customer letters | Presidio redaction for query, retrieved passages and response | Run 10-item synthetic PII suite and retain logs; pending |
| Measure | Citation and effective-date quality | Chain creates source citations and effective-date footer | Golden-set run; pending |
| Measure | Faithfulness, context precision/recall | Golden question set and CI evaluator scaffold | Automated judge report and single-variable reranker experiment; pending |
| Measure | Latency and false-positive cost | Layer timing fields exist in guardrail result objects | Capture median latency with/without each layer and false-positive rate; pending |
| Measure | Injection resistance | Prompt instructions, chunk screening adapter and input screening hook | Execute 3 poisoned-document attacks and retain results; pending |
| Manage | Fail safely on timeout, empty retrieval and guardrail trip | Typed refusal paths, 504 handler and structured API errors | API integration tests and failure-injection evidence; pending |
| Manage | Superseded documents are never presented as current | Version-aware ranking and explicit warning path | Test all 4 deliberately superseded versions; pending |
| Manage | Three independent guardrail layers | Adapter hooks exist; installed NeMo Guardrails and Guardrails AI are not yet configured/validated in this package | **Not yet demonstrated**: show one unique catch per NeMo, Guardrails AI and Llama Guard; report false positives and latency |
| Manage | Human review and residual-risk ownership | Human referral text present | Assign control owner and review cadence; pending |

## Release decision

Do not claim the case study meets the brief until the evidence above is attached, especially:
- >=90% golden questions answered with correct citation and effective date;
- 100% refusal on the advice test set;
- 0/10 synthetic PII records reaching a model call or trace;
- 0/3 poisoned documents changing behaviour or leaking data;
- guardrail comparison, latency and false-positive measurements;
- CI run passing the real prompt and failing a deliberately degraded prompt.
