from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern
from presidio_anonymizer import AnonymizerEngine

class PIIRedactor:
    def __init__(self):
        self.analyzer = AnalyzerEngine()
        self.anonymizer = AnonymizerEngine()

        #Custom pattern for Bank Accounts
        account_pattern = Pattern(
            name="account_number_pattern",
            regex=r"\b\d{8,12}\b",
            score=0.9
        )
        account_recognizer = PatternRecognizer(
            supported_entity="ACCOUNT_NUMBER",
            patterns=[account_pattern]
        )
        self.analyzer.registry.add_recognizer(account_recognizer)

        self.entities_to_redact = [
            "PERSON", 
            "PHONE_NUMBER",  
            "EMAIL_ADDRESS", 
            "LOCATION", 
            "ACCOUNT_NUMBER" # Our custom entity(Account number)
        ]

    def redact(self, text: str) -> str:
        if not text:
            return text

        results = self.analyzer.analyze(
            text=text,
            entities=self.entities_to_redact,
            language='en'
        )

        anonymized_result = self.anonymizer.anonymize(
            text=text,
            analyzer_results=results
        )
        
        return anonymized_result.text