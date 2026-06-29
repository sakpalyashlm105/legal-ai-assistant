# Legal AI Assistant

An agentic, LangGraph-based system for analyzing legal documents such as NDAs, contracts, supplier agreements, service agreements, and amendments.

The system extracts important clauses, compares them with reference templates, identifies risk, checks curated human-approved precedent, pauses for human review when necessary, and produces a structured final report.

This project was developed as an AI/ML engineering capstone with a deliberate focus on **safety-first architecture** rather than feature breadth. High-risk findings, missing clauses, and uncertain results remain subject to human review.

> **Disclaimer:** This project is an educational prototype. It does not provide legal advice and should not replace review by a qualified legal professional.

---

## Why this project exists

Reviewing contracts for missing, unusual, or non-standard clauses can be slow and error-prone when performed manually at scale.

The Legal AI Assistant automates the first review pass by handling:

* Document text extraction
* Document classification
* Clause identification
* Template comparison
* Risk assessment
* Evidence validation
* Human-review routing
* Structured report generation

The system does not make the final legal decision. A human reviewer remains the final authority for findings that are risky, missing, uncertain, or otherwise require judgment.

Over time, separately curated human-approved clause language may be used as precedent to reduce redundant review. Human feedback does not automatically update model weights or fine-tune the model.

---

## How it works

```text
Upload PDF
    │
    ▼
Extract text
(PyMuPDF with Vision OCR fallback for low-text pages)
    │
    ▼
Classify document type
(NDA / Contract / Amendment / Other)
    │
    ├──── Amendment detected?
    │          │
    │          ▼
    │     Amendment-summary path
    │     Detects modified clauses and summarizes changes
    │
    ▼
Extract the 10 locked clause categories
    │
    ▼
Compare present clauses against reference templates
    │
    ├── Confidence below 0.50
    │       Retry extraction or escalate
    │
    ├── Confidence between 0.50 and 0.70
    │       Use Tree-of-Thought reasoning
    │
    └── Confidence above 0.70
            Continue normally
    │
    ▼
Score clause risk
(Missing clause = HIGH risk)
    │
    ▼
Check curated precedent
    │
    ├── Similar human-approved language found?
    │       Apply only when permitted by safety rules
    │
    └── Missing or HIGH-risk finding?
            Do not downgrade using precedent
    │
    ▼
Run guardrails
(Input validation, prompt-injection scanning,
evidence verification, page validation,
claim validation, and final-output validation)
    │
    ▼
Pause for human review when required
    │
    ▼
Reviewer selects:
Approve / Correct / Reject / Select Alternative
    │
    ▼
Resume LangGraph workflow
    │
    ▼
Generate final Markdown and JSON report
```

---

## Core safety rule

### A missing clause is always treated as HIGH risk

A missing clause cannot be downgraded by precedent.

This rule is enforced at multiple levels:

1. A Pydantic schema validator prevents an unsafe missing-clause risk record from being created.
2. The precedent lookup contains a function-level safety check.
3. An upstream orchestration gate avoids precedent-based downgrading for absent clauses.
4. A permanent regression test, `REG-001`, protects this behavior from future changes.

The goal is not to claim that AI can never make an extraction error. The goal is to ensure that a detected missing clause cannot silently pass through the workflow as low risk.

---

## Human-in-the-loop design

This project uses **human-in-the-loop review**, not automatic reinforcement learning from feedback.

Human review records may capture:

* Whether the model finding was correct
* Whether the clause category was correct
* Whether the risk level was correct
* Whether the evidence was sufficient
* Whether the clause language is acceptable as business precedent
* Reviewer comments and corrections

Two decisions are intentionally kept separate:

1. **Accepting the model finding**
2. **Approving the clause language as reusable precedent**

A reviewer may confirm that a HIGH-risk finding is correct without approving that clause wording for future reuse.

Human decisions do not automatically:

* Update model weights
* Fine-tune the LLM
* Become precedent
* Change live routing thresholds

Precedent use requires a separate, explicit curation step.

---

## Review Score

The project includes a composite Review Score based on:

* Risk severity
* Extraction confidence
* Deviation from the reference template
* Evidence quality
* Finding novelty

The score is implemented, logged, and tested, but currently operates in **shadow mode**.

It does not control live routing or automatic approval.

This was a deliberate safety decision. The score requires additional calibration and real review data before it should be trusted to determine whether a finding can bypass human review.

Missing-clause findings receive a mandatory interrupt-floor score regardless of the individual component values.

---

## Technology stack

| Component      | Technology                                        |
| -------------- | ------------------------------------------------- |
| LLM            | OpenAI `gpt-4o-mini`                              |
| Embeddings     | OpenAI `text-embedding-3-small`                   |
| PDF extraction | PyMuPDF                                           |
| OCR fallback   | Vision-capable OpenAI model                       |
| Vector store   | FAISS                                             |
| Retrieval      | Similarity search, MMR, and LLM reranking         |
| Orchestration  | LangGraph                                         |
| Human review   | LangGraph `interrupt()` and `Command(resume=...)` |
| Validation     | Pydantic v2                                       |
| User interface | Streamlit                                         |
| Testing        | Pytest                                            |
| Reporting      | Markdown and JSON                                 |

---

## Project structure

```text
legal-ai-assistant/
├── agent/
│   ├── orchestrator.py
│   ├── classifier.py
│   ├── extractor.py
│   ├── comparator.py
│   ├── risk_engine.py
│   ├── review_score.py
│   ├── feedback_writer.py
│   ├── amendment_analyzer.py
│   ├── clause_expander.py
│   ├── missing_clause_verifier.py
│   └── tot_reasoner.py
│
├── extraction/
│   └── PDF extraction and OCR fallback
│
├── retrieval/
│   └── Chunking, embeddings, FAISS, and reranking
│
├── guardrails/
│   └── Input, scope, injection, evidence, page,
│       claim, and output validators
│
├── reporting/
│   └── Report generation and executive summaries
│
├── schemas/
│   └── Pydantic models used throughout the system
│
├── data/
│   ├── pdf/
│   ├── templates/
│   ├── feedback/
│   ├── metrics/
│   └── evaluation/
│
├── tests/
│   ├── unit/
│   └── integration/
│
├── main.py
├── demo_hitl.py
├── config.py
├── requirements.txt
├── .gitignore
├── README.md
└── CLAUDE.md
```

Generated feedback logs, metrics logs, local environment files, API keys, and sensitive documents should not be committed to the public repository.

---

## Installation

### 1. Clone the repository

```powershell
git clone https://github.com/sakpalyashlm105/legal-ai-assistant.git
cd legal-ai-assistant
```

### 2. Create a virtual environment

```powershell
python -m venv venv
```

### 3. Activate the virtual environment

Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Configure environment variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key
```

Never commit the `.env` file or a real API key to GitHub.

---

## Usage

### Streamlit application

```powershell
streamlit run main.py
```

Then:

1. Upload a supported PDF.
2. Start the analysis.
3. Review findings that trigger a human-review interrupt.
4. Approve, correct, reject, or select an alternative.
5. Resume the workflow.
6. Review the generated final report.

### Scripted human-review demonstration

```powershell
python demo_hitl.py
```

For a real PDF and live LLM call:

```powershell
python demo_hitl.py --real-llm --pdf "path\to\document.pdf"
```

Live LLM execution requires a valid OpenAI API key and may incur API charges.

---

## Testing

Run the complete test suite:

```powershell
python -m pytest tests/ -v --tb=short
```

Run only unit tests:

```powershell
python -m pytest tests/unit -v
```

Run the Review Score tests:

```powershell
python -m pytest tests/unit/test_review_score.py -v
```

Use:

```powershell
python -m pytest
```

instead of calling `pytest` directly if the local environment has a stale launcher or OneDrive-related path issue.

Do not publish an exact passing-test count unless it comes from a recent full-suite run.

---

## Known limitations

### Jurisdiction-blind precedent matching

Precedent matching currently relies primarily on clause category and text similarity.

Jurisdiction is not yet extracted and enforced as a strict compatibility requirement. Similar language from one jurisdiction could therefore match language from another jurisdiction.

### Limited amendment handling

The system detects amendments and routes them through an amendment-summary workflow.

It does not yet retrieve and cross-reference the original base agreement that the amendment modifies.

### Review Score remains in shadow mode

The Review Score is calculated and logged but does not affect live routing or automatic approval.

It requires calibration against a larger set of human-reviewed examples before production use should be considered.

### Retrieval is not yet connected to the live graph

The FAISS and embedding-based retrieval subsystem is implemented and tested independently.

A retrieval node is not yet connected to the main LangGraph execution path.

### Incomplete reference-template coverage

Reference templates currently exist for 5 of the 10 locked clause categories.

When a present clause has no template, the system has limited ability to evaluate deviation from a benchmark.

### Amendment analyzer test coverage

The amendment analyzer has been manually checked against real documents but still requires stronger automated unit and integration test coverage.

### Prototype status

The project is an engineering capstone and not a production legal-review platform.

Production deployment would require additional work involving:

* Legal-domain validation
* Access control
* Data encryption
* Audit retention policies
* Jurisdiction-aware logic
* Model and prompt versioning
* Monitoring
* Security review
* Calibration using representative legal documents
* Review by qualified legal subject-matter experts

---

## Roadmap

* Connect the FAISS retrieval subsystem to the live LangGraph workflow
* Build base-agreement retrieval for amendments
* Add templates for the remaining clause categories
* Extract and enforce governing jurisdiction
* Calibrate the Review Score using human-reviewed data
* Evaluate safe score-based routing
* Improve precedent governance and expiration controls
* Expand automated amendment tests
* Add stronger document-level evaluation datasets
* Add role-based access controls and production audit storage

---

## Repository

GitHub:

https://github.com/sakpalyashlm105/legal-ai-assistant

---

## License

No open-source license has been selected yet.

Until a license is added, the repository remains publicly viewable but does not automatically grant permission to copy, modify, or redistribute the code.
