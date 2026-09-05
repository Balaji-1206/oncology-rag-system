import urllib.request
import json
import time

questions = [
    ("1. Factual/Definition", "What is carcinoma and what are its primary histological subtypes?"),
    ("2. Treatment-related", "What are common first-line treatments for HER2-positive metastatic breast cancer?"),
    ("3. Comparison", "Compare the clinical features of small cell lung cancer versus non-small cell lung cancer."),
    ("4. Evidence-heavy/Staging", "In lung cancer staging, what is the clinical significance of mediastinal lymph node enlargement measuring 1.5 cm on a CT scan?"),
    ("5. Out-of-domain/Insufficient", "What are the standard chemotherapy protocols for automotive engine mechanical failure?")
]

results = []
for label, q in questions:
    print(f"\n==============================================\nRunning: {label}\nQuery: {q}")
    req_data = json.dumps({"query": q}).encode("utf-8")
    req = urllib.request.Request("http://127.0.0.1:5000/query", data=req_data, headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            elapsed = round(time.time() - t0, 2)
            eval_data = data.get("evaluation", {})
            metrics_data = data.get("metrics", {})
            api_metrics = data.get("api_metrics", {})
            qa = data.get("query_analysis", {})

            res_summary = {
                "label": label,
                "query": q,
                "status": "SUCCESS",
                "total_elapsed_s": elapsed,
                "api_request_time": api_metrics.get("request_time"),
                "intent": qa.get("intent"),
                "query_type": qa.get("query_type"),
                "keywords": qa.get("keywords", [])[:6],
                "retrieved_count": len(data.get("sources", [])),
                "top_sources": data.get("sources", [])[:3],
                "grounded": data.get("grounded"),
                "quality": data.get("quality"),
                "evaluation": {
                    "grounding_score": eval_data.get("grounding_score"),
                    "answer_relevance": eval_data.get("answer_relevance"),
                    "retrieval_score": eval_data.get("retrieval_score"),
                    "hallucination_risk": eval_data.get("hallucination_risk")
                },
                "stage_latencies": {
                    "laqa_time": metrics_data.get("laqa_time"),
                    "rag_time": metrics_data.get("rag_time"),
                    "xai_time": metrics_data.get("xai_time"),
                    "total_time": metrics_data.get("total_time")
                },
                "answer_preview": data.get("answer", "")[:250],
                "reasoning_preview": data.get("reasoning", "")[:200]
            }
            results.append(res_summary)
            print(f"Done in {elapsed}s | Grounded: {data.get('grounded')} | Risk: {eval_data.get('hallucination_risk')}")
            print(f"Answer Preview: {data.get('answer', '')[:120]}...")
    except Exception as e:
        print(f"FAILED: {e}")
        results.append({"label": label, "query": q, "status": f"FAILED: {e}"})

with open("backend/tests/e2e_validation_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print("\n[SUCCESS] All 5 E2E validation queries executed. Results saved to backend/tests/e2e_validation_results.json")
