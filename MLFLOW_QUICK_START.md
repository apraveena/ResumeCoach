# MLflow: Resume Coach quick start

1. Back up your current project. Extract this ZIP into the project root, replacing files. Keep your own `.env` and `GROQ_API_KEY`.
2. From the project root, install: `python -m pip install -r requirements.txt`
3. Smoke test: `python evals/eval_runner.py --judge --limit 3 --prompt-version v2`
4. Full comparison (API calls and costs apply):
   - `python evals/eval_runner.py --judge --prompt-version v1`
   - `python evals/eval_runner.py --judge --prompt-version v2`
5. Dashboard: `mlflow ui --backend-store-uri ./mlruns --host 0.0.0.0 --port 5000`
6. In Codespaces, open the forwarded port 5000 via the Ports tab, then open the `resume_coach_prompt_comparison` experiment and compare the two runs.

**Outputs:** Each run writes `evals/runs/<run_name>/results.csv` and `predictions.jsonl`. MLflow stores parameters, aggregate metrics, dataset and prompt SHA-256 hashes, and these artifacts in `mlruns/`.

**Metric caveats:** `matching_f1_mean` and `missing_f1_mean` exclude failed cases; always check `cases_error` and `judge_cases_error`. Judge means exclude cases with no rewrites. The LLM judge is subjective and needs human calibration. Run v1/v2 with the same case limit, judge model, and dataset. Groq usage may incur charges or hit rate limits. Do not commit real resumes or API keys to Git; this example includes only synthetic data.

**Troubleshooting:** Run commands from the project root. If `mlflow` command is missing, try `python -m mlflow ui --backend-store-uri ./mlruns --host 0.0.0.0 --port 5000`. If port 5000 is busy, use 5001. If a Groq model is unavailable, set `GROQ_JUDGE_MODEL` in `.env` to an available judge model.
