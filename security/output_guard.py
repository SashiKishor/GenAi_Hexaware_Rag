import os
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from Pii_Redactor import pii_redactor

class OutputAudit(BaseModel):
    is_grounded: bool
    is_compliant: bool
    reason: str

PROHIBITED_PHRASES = [
    "guaranteed approval",
    "100% guaranteed",
    "risk-free return",
    "we promise approval",
    "bypass verification",
]

DISCLAIMER = (
    "\n\nNote: Information is based on Northbridge Bank policy documents and "
    "is subject to official terms and eligibility criteria."
)

def _get_audit_llm():
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )
    return llm.with_structured_output(OutputAudit)

async def run_output_guardrail(llm_response: str, retrieved_chunks: list[str]) -> str:
    safe_response = pii_redactor.redact(llm_response)
    lowered = safe_response.lower()

    for phrase in PROHIBITED_PHRASES:
        if phrase in lowered:
            return (
                "I cannot make financial guarantees or promise approvals. "
                "Please consult Northbridge Bank's official policy schedule or a branch advisor."
            )

    if not retrieved_chunks:
        return (
            "I could not find verified information in Northbridge Bank's records "
            "to answer your request."
        )

    auditor = _get_audit_llm()
    context_text = "\n---\n".join(retrieved_chunks)
    prompt = (
        "You are a banking compliance auditor. Verify whether the Response is "
        "strictly supported by the Context without hallucinated rates, fees, or rules. "
        "Set is_grounded=True only if all factual claims appear in Context. "
        "Set is_compliant=True if it gives no unauthorized financial/legal promises.\n\n"
        f"Context:\n{context_text}\n\n"
        f"Response:\n{safe_response}"
    )

    audit: OutputAudit = await auditor.ainvoke(prompt)

    if not audit.is_grounded or not audit.is_compliant:
        return (
            "I could not verify all details of this answer against Northbridge Bank's "
            "official documentation. Please contact customer support for confirmation."
        )

    return safe_response + DISCLAIMER