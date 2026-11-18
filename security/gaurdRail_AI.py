#Optinal-Alternate for Pii_redactor-does the same work as an pii_redactor
# but can be used for both input and output-on output it enforces a format using pydantic models
# ex:an qoutation of the result should have an date in it and we can control what happens on fail

#pip install guardrails-ai-detect-pii

#libary did not work  

#do not use
from guardrails import Guard
# from guardrails_ai.detect_pii import DetectPII
from fastapi.exceptions import GuardrailTripException

# pii_guard = Guard().use(
#     DetectPII(
#         pii_entities=["PERSON", "PHONE_NUMBER", "EMAIL_ADDRESS", "US_BANK_NUMBER"],
#         on_fail="fix"
#     )
# )

# def validate_input(text: str) -> str:
#     try:
#         result = pii_guard.validate(text)
#         return result.validated_output
#     except Exception as e:
#         raise GuardrailTripException(f"Input validation failed: {str(e)}")

# text="My number is "

# print(validate_input(text))