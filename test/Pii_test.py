from unittest.mock import patch, MagicMock
from security.Pii_Redactor import PIIRedactor 


@patch("pii_redactor.AnonymizerEngine")
@patch("pii_redactor.AnalyzerEngine")
def test_redact_text(MockAnalyzer, MockAnonymizer):
    # Arrange
    redactor = PIIRedactor()
    redactor.analyzer.analyze.return_value = ["fake_pii_match"]
    redactor.anonymizer.anonymize.return_value = MagicMock(text="My name is Sashi And My account number is 021000021")

    # Act
    result = redactor.redact("My account is 1234567890")

    # Assert
    assert result == "My name is <PERSON> And My account number is <ACCOUNT_NUMBER>"
    redactor.analyzer.analyze.assert_called_once()
    redactor.anonymizer.anonymize.assert_called_once()