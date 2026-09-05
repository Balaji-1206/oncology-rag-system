import time
import traceback

import settings
from modules.laqa.laqa import process_query
from modules.agent.agent_controller import agent_decision
from modules.xai.explain import generate_explanation


def handle_query(user_query: str):
    """Core RAG pipeline handler executing LAQA, Retrieval, MedGemma, and XAI."""
    start_time = time.time()

    try:
        # Step 1: LAQA Query Preparation
        print("\n" + "─" * 66)
        print("  Step 1 · Query Preparation (LAQA)")
        print("─" * 66)

        laqa_start = time.time()
        if settings.is_rag_enabled() and settings.is_laqa_enabled():
            laqa_output = process_query(user_query)
        elif settings.is_rag_enabled():
            print("  ⚠  LAQA disabled: using raw user query.")
            laqa_output = settings.build_raw_query_payload(user_query)
        else:
            print("  ⚠  RAG disabled: bypassing LAQA and retrieval.")
            laqa_output = settings.build_raw_query_payload(user_query)

        laqa_time = round(time.time() - laqa_start, 2)

        # Step 2: Agentic RAG Execution
        print("\n" + "─" * 66)
        print("  Step 2 · Agent Execution (RAG + Retry Loop)")
        print("─" * 66)

        rag_start = time.time()
        agent_result = agent_decision(laqa_output)
        rag_time = round(time.time() - rag_start, 2)

        answer = agent_result.get("answer", "")
        docs = agent_result.get("docs", [])
        doc_ids = agent_result.get("doc_ids", [])
        eval_result = agent_result.get("eval", {})

        # Step 3: XAI Explainability Layer
        print("\n" + "─" * 66)
        print("  Step 3 · XAI (Explainability Layer)")
        print("─" * 66)

        xai_start = time.time()
        explanation = generate_explanation(answer, docs, eval_result, user_query)
        xai_time = round(time.time() - xai_start, 2)

        confidence = explanation.get("confidence", 0.65)
        if confidence is None or confidence <= 0:
            confidence = 0.65

        total_time = round(time.time() - start_time, 2)

        metrics = {
            "laqa_time": laqa_time,
            "rag_time": rag_time,
            "xai_time": xai_time,
            "total_time": total_time,
            "retrieval_score": eval_result.get("retrieval_score", 0),
            "grounding_score": eval_result.get("grounding_score", 0),
            "hallucination_risk": eval_result.get("hallucination_risk", "medium"),
            "answer_relevance": eval_result.get("answer_relevance", 0)
        }

        raw_answer = agent_result.get("raw_answer", answer)
        optimization_stats = agent_result.get("optimization_stats", {})

        return {
            "answer": answer,
            "raw_answer": raw_answer,
            "optimization_stats": optimization_stats,
            "confidence": confidence,
            "sources": doc_ids,
            "source_texts": docs,
            "context_docs": agent_result.get("context_docs", docs),
            "candidate_texts": agent_result.get("candidate_texts", []),
            "explanation": explanation,
            "evaluation": eval_result,
            "metrics": metrics,
            "disclaimer": (
                "For research and educational purposes only. "
                "Not certified for direct clinical diagnosis or medical decision support."
            ),
            "query_analysis": {
                "laqa_enabled": settings.is_laqa_enabled(),
                "intent": laqa_output.get("intent"),
                "query_type": laqa_output.get("query_type"),
                "expanded_query": laqa_output.get("expanded_query"),
                "keywords": laqa_output.get("keywords"),
                "query_metadata": laqa_output.get("query_metadata", {})
            }
        }

    except Exception as e:
        print("\n" + "═" * 66)
        print("  ❌ PIPELINE FAILURE")
        print("═" * 66)
        print(traceback.format_exc())

        return {
            "answer": "The oncology AI pipeline encountered an internal processing error.",
            "confidence": 0.1,
            "sources": [],
            "source_texts": [],
            "context_docs": [],
            "explanation": {
                "supporting_sentences": [],
                "reasoning": "Pipeline execution failed.",
                "confidence": 0.1,
                "quality": "Low",
                "grounded": False
            },
            "evaluation": {
                "score": 1,
                "needs_retry": False,
                "hallucination_risk": "medium"
            },
            "candidate_texts": [],
            "metrics": {
                "total_time": round(time.time() - start_time, 2)
            },
            "query_analysis": {},
            "error": str(e)
        }


if __name__ == "__main__":
    print("\n" + "═" * 66)
    print("  \033[1m🧠  Oncology AI Assistant  |  MRL 512-dim + BM25\033[0m")
    print("  \033[33m⚠️  RESEARCH & EDUCATIONAL USE ONLY — NOT FOR CLINICAL DIAGNOSIS\033[0m")
    print("═" * 66)
    print("  Type 'exit' to quit\n")

    while True:
        user_query = input("Enter your medical query: ").strip()

        if user_query.lower() == "exit":
            break

        result = handle_query(user_query)

        # Query Header
        qa = result.get("query_analysis", {})
        metrics = result.get("metrics", {})

        print("\n" + "═" * 66)
        print(f"  \033[1m❓ Query\033[0m   : {user_query}")
        print(f"  \033[1m🔍 Intent\033[0m  : {qa.get('intent', 'N/A')}  |  Type: \033[1m{qa.get('query_type', 'N/A')}\033[0m")

        expanded = qa.get('expanded_query', '')
        if expanded:
            print(f"  \033[1m📌 Expanded\033[0m: {expanded[:80]}")

        # Step 1: Query Expansion (LAQA)
        print("\n" + "─" * 66)
        print("  \033[1mStep 1 · Query Expansion (LAQA)\033[0m")
        print("─" * 66)

        kw = qa.get('keywords', [])
        if kw:
            kw_str = ", ".join(kw) if isinstance(kw, list) else str(kw)
            print(f"  \033[1mKeywords :\033[0m {kw_str}")

        meta = qa.get('query_metadata', {})
        if meta:
            cancer = meta.get('cancer_type', '')
            treatment = meta.get('treatment_type', '')
            if cancer:
                print(f"  \033[1mCancer   :\033[0m {cancer}")
            if treatment:
                print(f"  \033[1mTreatment:\033[0m {treatment}")

        # Step 3: Agent Answer & Response Optimization Layer
        raw_ans = result.get("raw_answer", "")
        opt_ans = result.get("answer", "")
        opt_stats = result.get("optimization_stats", {})

        print("\n" + "─" * 66)
        print("  \033[1mStep 3 · Response Synthesis & Optimization Layer\033[0m")
        print("─" * 66)

        print("\n  \033[1;33m📄 [1] Without Response Optimization (Raw LLM Output):\033[0m")
        print("  " + "┄" * 62)
        if raw_ans:
            for line in raw_ans.strip().splitlines():
                print(f"  {line}")
        else:
            print("  (No raw output recorded)")

        print("\n  \033[1;32m✨ [2] With Response Optimization (Refined Clinical Output):\033[0m")
        print("  " + "┄" * 62)
        if opt_ans:
            for line in opt_ans.strip().splitlines():
                print(f"  {line}")
        else:
            print("  (No optimized output available)")

        print("\n  \033[1;36m📊 [3] Optimization Diagnostics:\033[0m")
        print("  " + "┄" * 62)
        raw_c = opt_stats.get("raw_chars", len(raw_ans))
        opt_c = opt_stats.get("optimized_chars", len(opt_ans))
        red_p = opt_stats.get("reduction_percent", 0.0)
        art_s = opt_stats.get("artifacts_stripped", 0)
        dup_r = opt_stats.get("duplicate_lines_removed", 0)
        struct_t = opt_stats.get("structure_type", qa.get("query_type", "general"))

        ent_p = opt_stats.get("entities_polished", 0)

        print(f"  • Size Delta          : \033[1m{raw_c}\033[0m chars → \033[1m{opt_c}\033[0m chars ({red_p}% reduction)")
        print(f"  • Artifacts Stripped  : \033[1m{art_s}\033[0m pattern matches (<think>, prompt echoes, filler)")
        print(f"  • Duplicates Removed  : \033[1m{dup_r}\033[0m redundant lines")
        print(f"  • Entities Polished   : \033[1m{ent_p}\033[0m oncology terms/syntax standardizations")
        print(f"  • Structural Alignment: \033[1m{str(struct_t).capitalize()}\033[0m clinical formatting")

        # Step 4: Supporting Evidence
        supporting = result["explanation"].get("supporting_sentences", [])

        print("\n" + "─" * 66)
        print("  \033[1mStep 4 · Supporting Evidence\033[0m")
        print("─" * 66)

        if supporting:
            for i, s in enumerate(supporting, 1):
                print(f"  \033[1m[{i}]\033[0m \"{s}\"")
        else:
            print("  — No strong supporting evidence found.")

        # Step 5: XAI Explainability
        print("\n" + "─" * 66)
        print("  \033[1mStep 5 · XAI (Explainability)\033[0m")
        print("─" * 66)

        reasoning = result["explanation"].get("reasoning", "")
        if reasoning:
            for line in reasoning.strip().splitlines():
                print(f"  \033[1m•\033[0m {line.strip()}")
        else:
            print("  \033[1m•\033[0m No explanation available.")

        # Step 6: Quality Metrics
        print("\n" + "─" * 66)
        print("  \033[1mStep 6 · Quality Metrics\033[0m")
        print("─" * 66)

        eval_r = result.get("evaluation", {})
        confidence = result.get("confidence", 0)
        retrieval_score = eval_r.get("retrieval_score", 0)
        grounding = eval_r.get("grounding_score", 0)
        hallucination = eval_r.get("hallucination_risk", "N/A")
        answer_rel = eval_r.get("answer_relevance", 0)
        eval_score = eval_r.get("score", "N/A")
        needs_retry = eval_r.get("needs_retry", False)
        status = "\033[1m✅ ACCEPTED\033[0m" if not needs_retry else "\033[1m🔁 RETRIED\033[0m"

        print(
            f"  \033[1mConfidence     :\033[0m {confidence:.2f}      "
            f"\033[1mGrounding    :\033[0m {grounding:.2f}"
        )
        print(
            f"  \033[1mRetrieval Score:\033[0m {retrieval_score:.2f}      "
            f"\033[1mAnswer Rel.  :\033[0m {answer_rel:.2f}"
        )
        print(
            f"  \033[1mHallucination  :\033[0m {str(hallucination).upper()}       "
            f"\033[1mEvaluator    :\033[0m {eval_score}/10  {status}"
        )

        # Pipeline Timing Footer
        laqa_t = metrics.get('laqa_time', 0)
        rag_t = metrics.get('rag_time', 0)
        xai_t = metrics.get('xai_time', 0)
        total_t = metrics.get('total_time', 0)

        print(
            f"\n\033[1m⏱  LAQA:\033[0m {laqa_t}s  |  "
            f"\033[1mRAG:\033[0m {rag_t}s  |  "
            f"\033[1mXAI:\033[0m {xai_t}s  |  "
            f"\033[1mTotal:\033[0m {total_t}s"
        )
        print("═" * 66)
