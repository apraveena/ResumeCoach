"""Run golden-dataset checks against the existing coach.py. Run from project root."""
import csv
import argparse
import json
from pathlib import Path
import sys
import time
from datetime import datetime, timezone
from statistics import mean
import mlflow

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from coach import analyze_resume

DATASET = ROOT / "data" / "golden_dataset.json"
OUTPUT = ROOT / "evals" / "results.csv"


def normalized(items):
    return {" ".join(str(item).lower().replace("-", " ").split()) for item in items}


def prf(expected, actual):
    started = time.perf_counter()
    expected, actual = normalized(expected), normalized(actual)
    tp = len(expected & actual)
    precision = tp / len(actual) if actual else (1.0 if not expected else 0.0)
    recall = tp / len(expected) if expected else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def run(use_judge=False, limit=None, prompt_version="v2"):
    started = time.perf_counter()
    # Store tracking data in the project root, regardless of the launch directory.
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("resume_coach_prompt_comparison")
    dataset_text = DATASET.read_bytes()
    import hashlib
    dataset_sha = hashlib.sha256(dataset_text).hexdigest()
    # prompt_path = ROOT / "prompts" / f"prompts/prompt_{prompt_version}.txt"
    prompt_path = Path(__file__).resolve().parent.parent / "prompts" / f"prompt_{prompt_version}.txt"
    prompt_sha = hashlib.sha256(prompt_path.read_bytes()).hexdigest()
    from evals.llm_judge import JUDGE_PROMPT
    judge_sha = hashlib.sha256(JUDGE_PROMPT.encode()).hexdigest()
    run_name = f"{prompt_version}_{'judge' if use_judge else 'nojudge'}_{datetime.now(timezone.utc):'%Y%m%dT%H%M%SZ'}"
    model_name = "openai/gpt-oss-120b"
    judge_model = __import__("os").getenv("GROQ_JUDGE_MODEL", "openai/gpt-oss-20b")
    cases = json.loads(DATASET.read_text(encoding="utf-8"))
    if limit is not None:
        cases = cases[:limit]
    if not cases:
        raise ValueError("Golden dataset has no cases")
    rows = []
    predictions = []
    if use_judge:
        from evals.llm_judge import judge_rewrites
    for case in cases:
        row = {"id": case["id"], "status": "error", "matching_f1": 0.0,
               "missing_f1": 0.0, "score_in_range": False,
               "originals_grounded": False, "error": "",
               "judge_status": "not_run", "judge_avg_factual": "", "judge_avg_relevance": "",
               "judge_avg_star": "", "judge_avg_clarity": "", "judge_unsupported_count": "", "judge_error": ""}
        try:
            result = analyze_resume(case["resume"], case["job_description"], prompt_version=prompt_version)
            score = result["match_score"]
            assert isinstance(score, int) and not isinstance(score, bool) and 0 <= score <= 100
            assert isinstance(result["matching_skills"], list)
            assert isinstance(result["missing_skills"], list)
            assert isinstance(result["star_rewrites"], list)
            assert isinstance(result["summary"], str)
            assert isinstance(result["total_weak_bullets_found"], int)
            assert isinstance(result["rewrites_summary"], str)
            if prompt_version == "v2":
                assert isinstance(result["strengths_to_emphasize"], list)
                assert isinstance(result["coaching_actions"], list)
                interview = result["interview_preparation"]
                assert isinstance(interview, dict)
                assert all(isinstance(interview[key], list) for key in (
                    "likely_questions", "talking_points", "questions_to_ask"
                ))
            for rewrite in result["star_rewrites"]:
                assert all(isinstance(rewrite[k], str) for k in ("original", "rewritten", "reason"))
                if prompt_version == "v2":
                    assert type(rewrite["grounded"]) is bool
                    assert isinstance(rewrite["unsupported_claims"], list)
            row["matching_f1"] = round(prf(case["expected_matching_skills"], result["matching_skills"])[2], 3)
            row["missing_f1"] = round(prf(case["expected_missing_skills"], result["missing_skills"])[2], 3)
            row["score_in_range"] = case["score_range"][0] <= score <= case["score_range"][1]
            row["originals_grounded"] = all(item["original"] in case["resume"] for item in result["star_rewrites"])
            row["status"] = "pass" if row["score_in_range"] and row["originals_grounded"] else "review"
            record = {"id": case["id"], "prediction": result}
            predictions.append(record)
            if use_judge:
                try:
                    verdict = judge_rewrites(case["resume"], case["job_description"], result["star_rewrites"])
                    record["judge"] = verdict
                    items = verdict["items"]
                    row["judge_status"] = "ok"
                    if items:
                        for key, column in (("factual_accuracy", "judge_avg_factual"),
                                            ("relevance", "judge_avg_relevance"),
                                            ("star_structure", "judge_avg_star"),
                                            ("clarity", "judge_avg_clarity")):
                            row[column] = round(sum(x[key] for x in items) / len(items), 2)
                        row["judge_unsupported_count"] = sum(x["unsupported_claims"] for x in items)
                    else:
                        row["judge_unsupported_count"] = 0
                except Exception as judge_exc:
                    row["judge_status"] = "error"
                    row["judge_error"] = str(judge_exc)[:300]
        except Exception as exc:
            row["error"] = str(exc)[:300]
        rows.append(row)
        print(row)
    # Unique per-run filenames avoid overwriting older evaluation outputs.
    output_dir = ROOT / "evals" / "runs" / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"results_{prompt_version}.csv"
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    predictions_path = output_dir / f"predictions_{prompt_version}.jsonl"
    with predictions_path.open("w", encoding="utf-8") as fh:
        for record in predictions:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    valid_rows = [r for r in rows if r["status"] != "error"]
    judged_rows = [r for r in rows if r["judge_status"] == "ok" and r["judge_avg_factual"] != ""]
    metrics = {
        "cases_total": len(rows),
        "cases_success": len(valid_rows),
        "cases_error": len(rows) - len(valid_rows),
        "cases_review": sum(r["status"] == "review" for r in rows),
        "score_in_range_rate": sum(r["score_in_range"] for r in valid_rows) / len(valid_rows) if valid_rows else 0,
        "originals_grounded_rate": sum(r["originals_grounded"] for r in valid_rows) / len(valid_rows) if valid_rows else 0,
        "matching_f1_mean": mean(r["matching_f1"] for r in valid_rows) if valid_rows else 0,
        "missing_f1_mean": mean(r["missing_f1"] for r in valid_rows) if valid_rows else 0,
        "judge_cases_ok": sum(r["judge_status"] == "ok" for r in rows),
        "judge_cases_error": sum(r["judge_status"] == "error" for r in rows),
        "judge_unsupported_count": sum(int(r["judge_unsupported_count"]) for r in rows if r["judge_unsupported_count"] != ""),
        "duration_seconds": time.perf_counter() - started,
    }
    for col in ("judge_avg_factual", "judge_avg_relevance", "judge_avg_star", "judge_avg_clarity"):
        if judged_rows:
            metrics[col] = mean(float(r[col]) for r in judged_rows)
    with mlflow.start_run(run_name=run_name) as active_run:
        mlflow.log_params({
            "prompt_version": prompt_version,
            "coach_model": model_name,
            "judge_enabled": str(use_judge),
            "judge_model": judge_model if use_judge else "disabled",
            "judge_prompt_sha256": judge_sha if use_judge else "disabled",
            "dataset_sha256": dataset_sha,
            "prompt_sha256": prompt_sha,
            "case_limit": str(limit) if limit is not None else "all",
            "author_name": "Praveena & Buvana's Resume Coach Evaluation",
        })
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(output), artifact_path="evaluations")
        mlflow.log_artifact(str(predictions_path), artifact_path="evaluations")
        mlflow.log_artifact(str(prompt_path), artifact_path="prompts")
        mlflow.log_artifact(str(DATASET), artifact_path="dataset")
        print(f"MLflow run ID: {active_run.info.run_id}")
    print(f"Saved {output} and {predictions_path}")
    print("Mean matching F1 (successful cases):", round(metrics["matching_f1_mean"], 3))
    print("Mean missing F1 (successful cases):", round(metrics["missing_f1_mean"], 3))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge", action="store_true", help="Enable LLM judge API calls")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate first N cases")
    parser.add_argument("--prompt-version", choices=["v1", "v2"], default="v2")
    args = parser.parse_args()
    run(use_judge=args.judge, limit=args.limit, prompt_version=args.prompt_version)
