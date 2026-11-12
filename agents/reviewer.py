"""
reviewer.py
Reviewer Agent: Independently verifies and cross-checks numerical claims
and logic produced by the Analyst. If issues, hallucinations, or logic errors
are detected, formulates critique feedback and routes execution back to the Analyst
(up to a maximum retry threshold).
"""
