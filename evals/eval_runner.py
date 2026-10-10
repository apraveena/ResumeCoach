"""Run golden-dataset checks against the existing coach.py. Run from project root."""
import csv
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from coach import analyze_resume

DATASET = ROOT / "data" / "golden_dataset.json"
OUTPUT = ROOT / "evals" / "results.csv"


def normalized(items):
    return {" ".join(str(item).lower().replace("-", " ").split()) for item in items}


def prf(expected, actual):
    expected, actual = normalized(expected), normalized(actual)
    tp = len(expected & actual)
    precision = tp / len(actual) if actual else (1.0 if not expected else 0.0)
    recall = tp / len(expected) if expected else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def run(use_judge=False, limit=None, prompt_version="v2"):
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
            for rewrite in result["star_rewrites"]:
                assert all(isinstance(rewrite[k], str) for k in ("original", "rewritten", "reason"))
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
    output = ROOT / "evals" / f"results_{prompt_version}.csv"
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    predictions_path = ROOT / "evals" / f"predictions_{prompt_version}.jsonl"
    with predictions_path.open("w", encoding="utf-8") as fh:
        for record in predictions:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Saved {output} and {predictions_path}")
    print("Mean matching F1:", round(sum(r["matching_f1"] for r in rows) / len(rows), 3))
    print("Mean missing F1:", round(sum(r["missing_f1"] for r in rows) / len(rows), 3))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge", action="store_true", help="Enable LLM judge API calls")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate first N cases")
    parser.add_argument("--prompt-version", choices=["v1", "v2"], default="v2")
    args = parser.parse_args()
    run(use_judge=args.judge, limit=args.limit, prompt_version=args.prompt_version)
