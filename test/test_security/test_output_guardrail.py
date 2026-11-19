import pytest
from unittest.mock import AsyncMock, patch
from security.output_guard import (
    run_output_guardrail,
    OutputComplianceAudit
)


def create_audit(
    is_grounded=True,
    has_citation=True,
    quotes_fee_or_number=False,
    states_effective_date=True,
    uses_superseded_without_warning=False,
    gives_restricted_advice=False,
    reason="Test"
):
    return OutputComplianceAudit(
        is_grounded=is_grounded,
        has_citation=has_citation,
        quotes_fee_or_number=quotes_fee_or_number,
        states_effective_date=states_effective_date,
        uses_superseded_without_warning=uses_superseded_without_warning,
        gives_restricted_advice=gives_restricted_advice,
        reason=reason
    )


@pytest.fixture
def retrieved_chunks():
    return [
        {
            "document_id": "Northbridge_Fee_Schedule",
            "version": "2.0",
            "effective_date": "2026-01-01",
            "status": "current",
            "page": 5,
            "content": "The monthly maintenance fee is ₹500."
        }
    ]


@pytest.mark.asyncio
async def test_empty_retrieval():
    result = await run_output_guardrail(
        "The fee is ₹500.",
        []
    )

    assert result.passed is False
    assert result.tripped_layer == "empty_retrieval_guard"


@pytest.mark.asyncio
async def test_valid_grounded_response(retrieved_chunks):
    mock_auditor = AsyncMock()
    mock_auditor.ainvoke.return_value = create_audit()

    with patch(
        "security.output_guard._get_guardrails_ai_auditor",
        return_value=mock_auditor
    ):
        result = await run_output_guardrail(
            "The monthly maintenance fee is ₹500. "
            "Source: Northbridge_Fee_Schedule, Version 2.0, Page 5. "
            "Effective date: 2026-01-01.",
            retrieved_chunks
        )

    assert result.passed is True
    assert result.tripped_layer is None


@pytest.mark.asyncio
async def test_ungrounded_response_block(retrieved_chunks):
    mock_auditor = AsyncMock()
    mock_auditor.ainvoke.return_value = create_audit(
        is_grounded=False
    )

    with patch(
        "security.output_guard._get_guardrails_ai_auditor",
        return_value=mock_auditor
    ):
        result = await run_output_guardrail(
            "The bank provides a free account.",
            retrieved_chunks
        )

    assert result.passed is False
    assert result.tripped_layer == "hallucination_block"


@pytest.mark.asyncio
async def test_missing_citation_block(retrieved_chunks):
    mock_auditor = AsyncMock()
    mock_auditor.ainvoke.return_value = create_audit(
        has_citation=False
    )

    with patch(
        "security.output_guard._get_guardrails_ai_auditor",
        return_value=mock_auditor
    ):
        result = await run_output_guardrail(
            "The monthly maintenance fee is ₹500.",
            retrieved_chunks
        )

    assert result.passed is False
    assert result.tripped_layer == "citation_check"


@pytest.mark.asyncio
async def test_fee_without_effective_date_block(retrieved_chunks):
    mock_auditor = AsyncMock()
    mock_auditor.ainvoke.return_value = create_audit(
        quotes_fee_or_number=True,
        states_effective_date=False
    )

    with patch(
        "security.output_guard._get_guardrails_ai_auditor",
        return_value=mock_auditor
    ):
        result = await run_output_guardrail(
            "The monthly maintenance fee is ₹500.",
            retrieved_chunks
        )

    assert result.passed is False
    assert result.tripped_layer == "effective_date_check"


@pytest.mark.asyncio
async def test_restricted_advice_block(retrieved_chunks):
    mock_auditor = AsyncMock()
    mock_auditor.ainvoke.return_value = create_audit(
        gives_restricted_advice=True
    )

    with patch(
        "security.output_guard._get_guardrails_ai_auditor",
        return_value=mock_auditor
    ):
        result = await run_output_guardrail(
            "You should invest in this fund.",
            retrieved_chunks
        )

    assert result.passed is False
    assert result.tripped_layer == "restricted_advice_block"


@pytest.mark.asyncio
async def test_superseded_policy_warning(retrieved_chunks):
    mock_auditor = AsyncMock()
    mock_auditor.ainvoke.return_value = create_audit(
        uses_superseded_without_warning=True
    )

    with patch(
        "security.output_guard._get_guardrails_ai_auditor",
        return_value=mock_auditor
    ):
        result = await run_output_guardrail(
            "The monthly maintenance fee is ₹500.",
            retrieved_chunks
        )

    assert result.passed is True
    assert "SUPERSEDED" in result.final_answer


@pytest.mark.asyncio
async def test_pii_redacted_from_response(retrieved_chunks):
    mock_auditor = AsyncMock()
    mock_auditor.ainvoke.return_value = create_audit()

    with patch(
        "security.output_guard._get_guardrails_ai_auditor",
        return_value=mock_auditor
    ):
        result = await run_output_guardrail(
            "Customer Manojkanna's phone number is 9876543210. "
            "The monthly maintenance fee is ₹500. "
            "Source: Northbridge_Fee_Schedule, Version 2.0, Page 5. "
            "Effective date: 2026-01-01.",
            retrieved_chunks
        )

    assert result.passed is True
    assert "9876543210" not in result.final_answer
    assert "Manojkanna" not in result.final_answer


@pytest.mark.asyncio
async def test_auditor_called(retrieved_chunks):
    mock_auditor = AsyncMock()
    mock_auditor.ainvoke.return_value = create_audit()

    with patch(
        "security.output_guard._get_guardrails_ai_auditor",
        return_value=mock_auditor
    ):
        await run_output_guardrail(
            "The monthly maintenance fee is ₹500. "
            "Source: Northbridge_Fee_Schedule, Version 2.0, Page 5. "
            "Effective date: 2026-01-01.",
            retrieved_chunks
        )

    mock_auditor.ainvoke.assert_awaited_once()