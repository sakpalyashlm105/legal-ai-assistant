# Limitations — Legal AI Assistant

This document records known limitations of the current system that reviewers,
graders, and future developers should be aware of.

---

## Template Coverage Limitations

Template-based comparison is currently available for 5 of the 10
approved clause categories:
- Confidentiality / Non-Disclosure
- Indemnification
- Governing Law / Jurisdiction
- Limitation of Liability
- Termination for Convenience

The following 5 categories have NO template file in `data/templates/`
and therefore never undergo template comparison, even when the clause
is present in a document:
- Termination for Cause
- Non-Compete / Non-Solicitation
- Assignment
- Renewal / Term
- Dispute Resolution

**Practical effect**: for these 5 categories, a present clause will
never be flagged as a template deviation or receive a HIGH/MEDIUM risk
level based on non-standard language, regardless of how unusual its
terms are. This is a deliberate, conservative design choice (comparison
failures default to matches_template=True rather than manufacturing
false risk findings), but it means risk coverage is currently
asymmetric across categories. Adding template files for the remaining
5 categories is recommended future work before this system is relied
upon for comprehensive risk coverage.
