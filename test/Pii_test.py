import pytest
from security.Pii_Redactor import PIIRedactor 

@pytest.fixture(scope="module")
def redactor():
    return PIIRedactor()


def test_redact_person_and_account_number(redactor):
    text = "My name is Sashi And My account number is 021000021"

    result = redactor.redact(text)

    assert "021000021" not in result
    assert "<ACCOUNT_NUMBER>" in result
    assert "Sashi" not in result
    assert "<PERSON>" in result


def test_redact_email_and_phone(redactor):
    text = "Contact john.doe@example.com or 555-123-4567"

    result = redactor.redact(text)

    assert "john.doe@example.com" not in result
    assert "<EMAIL_ADDRESS>" in result
    assert "555-123-4567" not in result
    assert "<PHONE_NUMBER>" in result


def test_redact_empty_text(redactor):
    assert redactor.redact("") == ""