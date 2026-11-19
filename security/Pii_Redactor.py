from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern
from presidio_anonymizer import AnonymizerEngine


class PIIRedactor:
    def __init__(self):
        self.analyzer = AnalyzerEngine()
        self.anonymizer = AnonymizerEngine()

        recognizers = [
            PatternRecognizer(
                supported_entity="ACCOUNT_NUMBER",
                patterns=[
                    Pattern(
                        name="account_number",
                        regex=r"\b\d{8,18}\b",
                        score=0.9
                    )
                ]
            ),
            PatternRecognizer(
                supported_entity="IN_PAN",
                patterns=[
                    Pattern(
                        name="pan_number",
                        regex=r"\b[A-Z]{5}\d{4}[A-Z]\b",
                        score=0.95
                    )
                ]
            ),
            PatternRecognizer(
                supported_entity="IN_AADHAAR",
                patterns=[
                    Pattern(
                        name="aadhaar_number",
                        regex=r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}\b",
                        score=0.95
                    )
                ]
            ),
            PatternRecognizer(
                supported_entity="NATIONAL_ID",
                patterns=[
                    Pattern(
                        name="national_id",
                        regex=r"\b[A-Z]{2}\d{6,10}\b",
                        score=0.85
                    )
                ]
            ),
            PatternRecognizer(
                supported_entity="PERSON",
                patterns=[
                    Pattern(
                        name="person_name_phrase",
                        regex=r"(?i)(?<=my name is )[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*|(?<=customer )[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*",
                        score=0.85
                    )
                ]
            )
        ]

        for recognizer in recognizers:
            self.analyzer.registry.add_recognizer(recognizer)

        self.entities_to_redact = [
            "PERSON",
            "PHONE_NUMBER",
            "EMAIL_ADDRESS",
            "LOCATION",
            "CREDIT_CARD",
            "IN_PAN",
            "IN_AADHAAR",
            "ACCOUNT_NUMBER",
            "NATIONAL_ID"
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