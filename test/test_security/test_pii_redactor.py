import pytest
from security.Pii_Redactor import pii_redactor


def test_redact_person_name():
    text = "My name is Manojkanna."
    result = pii_redactor.redact(text)

    assert "Manojkanna" not in result


def test_redact_email():
    text = "Contact me at manoj@example.com"
    result = pii_redactor.redact(text)

    assert "manoj@example.com" not in result


def test_redact_phone_number():
    text = "My phone number is 9876543210"
    result = pii_redactor.redact(text)

    assert "9876543210" not in result


def test_redact_account_number():
    text = "Account number is 123456789012"
    result = pii_redactor.redact(text)

    assert "123456789012" not in result


def test_redact_pan_number():
    text = "My PAN number is ABCDE1234F"
    result = pii_redactor.redact(text)

    assert "ABCDE1234F" not in result


def test_redact_aadhaar_number():
    text = "My Aadhaar number is 1234 5678 9012"
    result = pii_redactor.redact(text)

    assert "1234 5678 9012" not in result
    assert "123456789012" not in result


def test_redact_multiple_pii():
    text = (
        "My name is Manojkanna. "
        "Email manoj@example.com. "
        "Phone 9876543210. "
        "Account 123456789012."
    )

    result = pii_redactor.redact(text)

    assert "Manojkanna" not in result
    assert "manoj@example.com" not in result
    assert "9876543210" not in result
    assert "123456789012" not in result


def test_empty_string():
    assert pii_redactor.redact("") == ""


def test_none_input():
    assert pii_redactor.redact(None) is None


def test_normal_text_remains():
    text = "What are the banking services available?"
    result = pii_redactor.redact(text)

    assert result == text