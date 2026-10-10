"""Run golden-dataset checks against the existing coach.py. Run from project root."""
import csv
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


def run():
    cases = json.loads(DATASET.read_text(encoding="utf-8"))
    rows = []
    for case in cases:
        row = {"id": case["id"], "status": "error", "matching_f1": 0.0,
               "missing_f1": 0.0, "score_in_range": False,
               "originals_grounded": False, "error": ""}
        try:
            result = analyze_resume(case["resume"], case["job_description"])
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
        except Exception as exc:
            row["error"] = str(exc)[:300]
        rows.append(row)
        print(row)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {OUTPUT}")
    print("Mean matching F1:", round(sum(r["matching_f1"] for r in rows) / len(rows), 3))
    print("Mean missing F1:", round(sum(r["missing_f1"] for r in rows) / len(rows), 3))


if __name__ == "__main__":
    run()
