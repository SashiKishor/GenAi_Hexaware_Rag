# Northbridge Bank RAG Assistant

Internal synthetic-data case-study implementation. Python 3.11 is recommended.

## Setup

1. Create a virtual environment and install `requirements.txt`.
2. Copy `.env.example` to `.env`; set `GOOGLE_API_KEY`. Never commit `.env`.
3. Ensure the synthetic PDFs are present under `data/raw/`.
4. Run multimodal ingestion: `python -m multimodal_rag.multimodal_ingestion`.
5. Start the API: `uvicorn app.api:app --reload`.
6. Register a user, log in, then call authenticated `POST /ask` or `POST /ask/stream`.

The multimodal API path combines version-aware text retrieval with table/chart/diagram retrieval. Retrieved text is PII-redacted and screened before prompt construction. Llama Guard is opt-in because its weights may be large and may require Hugging Face access; set `ENABLE_LLAMA_GUARD=true` only after obtaining the model and validating it locally.

## Configuration

- `GOOGLE_API_KEY`: required for Gemini chat, input classification and output audit.
- `GEMINI_CHAT_MODEL`: defaults to `gemini-2.5-flash`.
- `MIN_RELEVANCE`: defaults to `0.15`; calibrate against the golden set before release.
- `ENABLE_LLAMA_GUARD`: defaults to `false`.
- `LLAMA_GUARD_MODEL`: defaults to `meta-llama/Llama-Guard-3-1B`.

## Tests and evaluation

- Syntax check: `python -m compileall -q app rag multimodal_rag security`
- Unit tests: `pytest -q`
- Golden-set gate: save actual, redacted API outputs to `evaluation/results.json`, then run `python evaluation/evaluate.py --results evaluation/results.json`.

The evaluator checks citation/effective-date coverage and the investment-advice refusal. Faithfulness, precision/recall, latency, costs, PII 0/10, injection 0/3, and the unique-catch comparison for all three named guardrail products still require real experiment runs. See `reports/`.

## Security notes

- `.env` and local database files are excluded from the delivered archive.
- Rotate any credential that was present in the previously shared archive.
- The package currently wires a Gemini-based input classifier and output auditor plus optional Llama Guard. **NeMo Guardrails and Guardrails AI are listed as dependencies but are not yet configured and demonstrated as independent runtime layers. F8 is therefore not complete.** Do not represent the three-layer requirement as passed until the evidence table shows each layer uniquely catching a failure.
