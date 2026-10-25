# ResumeCoach

## Project Purpose

ResumeCoach is an AI-powered resume coaching application. It compares a candidate's resume with a target job description and produces practical, evidence-based feedback.

The application is designed to help a job seeker:

- understand how well the resume matches the target role;
- identify skills that are present and skills that may be missing;
- improve vague resume bullet points using the STAR framework; and
- download a tailored resume and a plain-text coaching report.

This project demonstrates document processing, prompt-based large language model use, structured JSON validation, file generation, and evaluation of AI output.

## How the Application Works

```text
Resume file + job description
		 |
		 v
	 Resume text extraction
		 |
		 v
	 Groq AI resume analysis
		 |
		 v
  Match score, skill gaps, and rewrites
		 |
		 v
   Tailored DOCX and coaching report
```

The application uses prompt version 2 by default. This prompt emphasizes source grounding: the model should improve wording without inventing metrics, employers, qualifications, or achievements.

## Project Structure

| File or folder | Purpose |
|---|---|
| `app.py` | Streamlit user interface and end-to-end application flow |
| `parser.py` | Extracts readable text from PDF and DOCX resumes |
| `coach.py` | Calls the Groq model and parses the structured AI response |
| `coach_v1.py` | Earlier standalone version of the coaching prompt and call |
| `doc_generator.py` | Creates tailored DOCX and text report downloads |
| `prompts/` | Versioned system prompts used by the coach |
| `data/golden_dataset.json` | Expected test cases for skill matching and score ranges |
| `evals/eval_runner.py` | Runs automated evaluation against the golden dataset |
| `evals/llm_judge.py` | Independently reviews the quality and grounding of rewrites |
| `evals/runs/` | Saved evaluation results and model predictions |
| `MLFLOW_QUICK_START.md` | Instructions for comparing prompt versions with MLflow |

## Setup

From the project root, install the dependencies:

```bash
python -m pip install -r requirements.txt
```

Create a `.env` file and add a Groq API key:

```env
GROQ_API_KEY=your_key_here
```

The `.env` file contains a secret and must not be included in a submission or committed to Git.

## Run the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

The interface accepts a text-searchable PDF or DOCX resume. Scanned image-only resumes are rejected because the parser cannot extract reliable text from them. The job description can be pasted directly or fetched from a URL.

## Run the Evaluation

Run a three-case smoke test:

```bash
python evals/eval_runner.py --judge --limit 3 --prompt-version v2
```

Run the complete golden-dataset evaluation:

```bash
python evals/eval_runner.py --judge --prompt-version v2
```

The evaluator checks:

- whether the AI response follows the required JSON schema;
- whether the match score is within the expected range;
- precision/recall/F1-style skill matching results;
- whether rewritten source text exists in the original resume; and
- rewrite quality, relevance, STAR structure, clarity, and unsupported claims.

The LLM judge is advisory. Its results should be reviewed by a person because language-model grading is subjective. Evaluation commands make live Groq API calls and may incur usage costs.

To compare the two prompt versions:

```bash
python evals/eval_runner.py --judge --prompt-version v1
python evals/eval_runner.py --judge --prompt-version v2
```

## Known Limitations

- The application requires a Groq API key and an internet connection for AI analysis.
- PDF parsing works for text-based PDFs, not scanned images.
- Job-description extraction from websites depends on the website allowing automated requests and exposing readable page text.
- The match score is an estimate from the language model, not an official ATS score.
- The AI must not invent resume facts; the evaluation judge is used to identify cases where a rewrite may violate this rule.
- The current project is a Streamlit MVP and does not include user accounts, persistent resume storage, or production deployment configuration.

## Evaluation Result

The saved full evaluation contains 30 golden test cases. The recorded run passed the structural, score-range, and source-grounding checks for all cases. The evaluation also showed that rewrite grounding remains the main area for improvement, which is why prompt version 2 explicitly prohibits invented numbers and achievements.

## Academic Scope

This project demonstrates how an AI application can combine a user interface, document parsing, prompt engineering, structured output handling, generated documents, and reproducible evaluation. The evaluation artifacts are included so the behavior of different prompt versions can be compared rather than judged only by informal examples.