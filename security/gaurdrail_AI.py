#from guardrails import Guard
#from guardrails_ai.detect_pii import DetectPII

# Initialize the Guard with the PII entities you want to redact
# pii_guard = Guard().use(
#     DetectPII(
#         pii_entities=["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "CREDIT_CARD"],
#         on_fail="fix",
#     )
# )

# def sanitize_prompt(user_prompt: str) -> str:
#     result = pii_guard.validate(user_prompt)
#     return result.validated_output

# Example usage before sending the prompt to your LLM
# raw_prompt = "Hi, I'm Rahul Sharma. Email me at rahul.sharma@example.com or call +91-9876543210."
# clean_prompt = sanitize_prompt(raw_prompt)

# print("Sanitized Prompt:", clean_prompt)

#alternate for the PII redaction

#Has more functionality than Pii redaction(Persidio)

#Major fuctionality - on_fail="fix"(On failure we can tell the model what to do)