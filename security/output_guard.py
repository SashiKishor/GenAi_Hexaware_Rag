import os
import time
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from security.Pii_Redactor import pii_redactor


class OutputComplianceAudit(BaseModel):
    is_grounded: bool
    has_citation: bool
    quotes_fee_or_number: bool
    states_effective_date: bool
    uses_superseded_without_warning: bool
    gives_restricted_advice: bool
    reason: str


class OutputGuardResult(BaseModel):
    passed: bool
    final_answer: str
    tripped_layer: str | None = None
    latency_ms: float = 0.0


def _get_guardrails_ai_auditor():
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )

    return llm.with_structured_output(OutputComplianceAudit)


async def run_output_guardrail(
    llm_response: str,
    retrieved_chunks: list[dict]
) -> OutputGuardResult:

    t0 = time.perf_counter()

    if not retrieved_chunks:
        return OutputGuardResult(
            passed=False,
            final_answer=(
                "I cannot answer this question because no supporting "
                "Northbridge Bank policy or fee document was found in the corpus."
            ),
            tripped_layer="empty_retrieval_guard",
            latency_ms=round((time.perf_counter() - t0) * 1000, 2),
        )

    safe_response = pii_redactor.redact(llm_response)

    formatted_context = "\n---\n".join(
        f"[Doc: {chunk.get('document_id', 'unknown')} | "
        f"Version: {chunk.get('version', 'unknown')} | "
        f"Effective Date: {chunk.get('effective_date', 'unknown')} | "
        f"Status: {chunk.get('status', 'current')} | "
        f"Page: {chunk.get('page', 'N/A')}]\n"
        f"{chunk.get('content', '')}"
        for chunk in retrieved_chunks
    )

    auditor = _get_guardrails_ai_auditor()

    prompt = (
        "You are the Output Compliance Validator for Northbridge Bank.\n\n"
        "Audit the Draft Response against the Retrieved Context.\n\n"
        "1. is_grounded: True only if every factual claim, fee, threshold, "
        "KYC step, and rule in the response is explicitly supported by "
        "Retrieved Context.\n"
        "2. has_citation: True only if the response includes an identifiable "
        "source document and version or page/section citation.\n"
        "3. quotes_fee_or_number: True if the response mentions a fee, rate, "
        "balance threshold, percentage, or numeric limit.\n"
        "4. states_effective_date: True if the response explicitly states "
        "the effective date of the quoted policy or fee.\n"
        "5. uses_superseded_without_warning: True if the context contains "
        "superseded information and the response fails to warn the reader.\n"
        "6. gives_restricted_advice: True if the response provides investment "
        "recommendations, tax strategies, or credit approval decisions.\n\n"
        f"Retrieved Context:\n{formatted_context}\n\n"
        f"Draft Response:\n{safe_response}"
    )

    audit: OutputComplianceAudit = await auditor.ainvoke(prompt)

    elapsed = round(
        (time.perf_counter() - t0) * 1000,
        2
    )

    if audit.gives_restricted_advice:
        return OutputGuardResult(
            passed=False,
            final_answer=(
                "I cannot provide investment, tax, or credit-approval advice. "
                "Please refer this inquiry to the Northbridge Wealth & "
                "Investment Advisory Desk (ext. 4410)."
            ),
            tripped_layer="restricted_advice_block",
            latency_ms=elapsed,
        )

    if not audit.is_grounded:
        return OutputGuardResult(
            passed=False,
            final_answer=(
                "Refused: The generated response contained unverified "
                "figures or claims not supported by Northbridge Bank's "
                "retrieved documents."
            ),
            tripped_layer="hallucination_block",
            latency_ms=elapsed,
        )

    if not audit.has_citation:
        return OutputGuardResult(
            passed=False,
            final_answer=(
                "Refused: Compliance policy prohibits returning an answer "
                "without an explicit source citation."
            ),
            tripped_layer="citation_check",
            latency_ms=elapsed,
        )

    if audit.quotes_fee_or_number and not audit.states_effective_date:
        return OutputGuardResult(
            passed=False,
            final_answer=(
                "Refused: Any quoted fee or threshold must include its "
                "official effective date. Please re-query specifying the "
                "product tier or consult the latest Fee Schedule."
            ),
            tripped_layer="effective_date_check",
            latency_ms=elapsed,
        )

    if audit.uses_superseded_without_warning:
        safe_response = (
            "WARNING: This answer relies on a SUPERSEDED policy version "
            "that is no longer current.\n\n"
            + safe_response
        )

    return OutputGuardResult(
        passed=True,
        final_answer=safe_response,
        latency_ms=elapsed,
    )