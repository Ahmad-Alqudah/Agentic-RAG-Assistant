import json
from datetime import datetime, timezone
from pathlib import Path

from app.agents.agent_graph import run_agent


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "evaluation" / "results.json"


TEST_CASES = [
    {
        "question": "What is the work from home policy?",
        "expected_status": "answered",
        "expected_keywords": ["two", "days", "manager"],
    },
    {
        "question": "What does SEC-204 mean?",
        "expected_status": "answered",
        "expected_keywords": ["unauthorized", "access"],
    },
    {
        "question": "What does P1 indicate?",
        "expected_status": "answered",
        "expected_keywords": ["production", "outage"],
    },
    {
        "question": "Who is the CEO of NovaTech?",
        "expected_status": "abstained",
        "expected_keywords": [],
    },
    {
        "question": "What is NovaTech's stock price?",
        "expected_status": "abstained",
        "expected_keywords": [],
    },
    {
        "question": "What is the company founder's birthday?",
        "expected_status": "abstained",
        "expected_keywords": [],
    },
    {
        "question": "What should I do about a security incident?",
        "expected_status": "human_review",
        "expected_keywords": [],
    },
    {
        "question": "How do I report phishing?",
        "expected_status": "human_review",
        "expected_keywords": [],
    },
    {
        "question": "Can I disable MFA?",
        "expected_status": "human_review",
        "expected_keywords": [],
    },
]


def evaluate():
    results = []

    for index, case in enumerate(TEST_CASES, start=1):
        question = case["question"]

        print(f"\n[{index}/{len(TEST_CASES)}] {question}")

        try:
            output = run_agent(question)

            status = output["status"]
            answer = output["answer"]

            status_passed = (
                status == case["expected_status"]
            )

            keywords = case["expected_keywords"]

            keyword_passed = (
                all(
                    keyword.lower() in answer.lower()
                    for keyword in keywords
                )
                if keywords
                else True
            )

            passed = status_passed and keyword_passed

            result = {
                "question": question,
                "expected_status": case["expected_status"],
                "actual_status": status,
                "status_passed": status_passed,
                "keyword_passed": keyword_passed,
                "passed": passed,
                "answer": answer,
                "retry_count": output["retry_count"],
                "sources": output["retrieved_sources"],
            }

        except Exception as error:
            result = {
                "question": question,
                "expected_status": case["expected_status"],
                "actual_status": "error",
                "status_passed": False,
                "keyword_passed": False,
                "passed": False,
                "error": repr(error),
            }

        results.append(result)

        print("EXPECTED:", case["expected_status"])
        print("ACTUAL:", result["actual_status"])
        print("RESULT:", "PASS" if result["passed"] else "FAIL")

    total = len(results)
    passed_count = sum(item["passed"] for item in results)
    status_passed_count = sum(
        item["status_passed"] for item in results
    )

    summary = {
        "timestamp_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "total": total,
        "passed": passed_count,
        "failed": total - passed_count,
        "pass_rate": round(passed_count / total, 4),
        "routing_accuracy": round(
            status_passed_count / total,
            4,
        ),
    }

    payload = {
        "summary": summary,
        "results": results,
    }

    OUTPUT_PATH.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 55)
    print("FINAL EVALUATION")
    print("=" * 55)
    print("TOTAL:", total)
    print("PASSED:", passed_count)
    print("FAILED:", total - passed_count)
    print("PASS RATE:", f"{summary['pass_rate'] * 100:.1f}%")
    print(
        "ROUTING ACCURACY:",
        f"{summary['routing_accuracy'] * 100:.1f}%",
    )
    print("SAVED:", OUTPUT_PATH)

    return summary


if __name__ == "__main__":
    evaluate()