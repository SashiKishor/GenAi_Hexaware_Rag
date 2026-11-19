from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern
from presidio_anonymizer import AnonymizerEngine

class PIIRedactor:
    def __init__(self):
        self.analyzer = AnalyzerEngine()
        self.anonymizer = AnonymizerEngine()

        account_pattern = Pattern(
            name="account_number_pattern",
            regex=r"\b\d{8,18}\b",
            score=0.9
        )
        national_id_pattern = Pattern(
            name="national_id_pattern",
            regex=r"\b(?:[A-Z]{5}\d{4}[A-Z]|\d{4}[\s-]?\d{4}[\s-]?\d{4}|[A-Z]{2}\d{6,10})\b",
            score=0.85
        )
        self.analyzer.registry.add_recognizer(
            PatternRecognizer(supported_entity="ACCOUNT_NUMBER", patterns=[account_pattern])
        )
        self.analyzer.registry.add_recognizer(
            PatternRecognizer(supported_entity="NATIONAL_ID", patterns=[national_id_pattern])
        )

        self.entities_to_redact = [
            "PERSON",
            "PHONE_NUMBER",
            "EMAIL_ADDRESS",
            "LOCATION",
            "CREDIT_CARD",
            "IN_PAN",
            "IN_AADHAAR",
            "ACCOUNT_NUMBER",
            "NATIONAL_ID",
        ]

    def redact(self, text: str) -> str:
        if not text:
            return text
        results = self.analyzer.analyze(
            text=text,
            entities=self.entities_to_redact,
            language="en"
        )
        return self.anonymizer.anonymize(
            text=text,
            analyzer_results=results
        ).text

pii_redactor = PIIRedactor()