import time
import traceback

import settings
from modules.laqa.laqa import process_query

from modules.agent.agent_controller import (
    agent_decision
)

from modules.xai.explain import (
    generate_explanation
)


# =========================================================
# 🔹 CORE PIPELINE
# =========================================================
def handle_query(user_query: str):

    start_time = time.time()

    try:

        # =====================================================
        # 🔹 STEP 1: LAQA
        # =====================================================
        print("\n🔹 Step 1: Query Preparation...")

        laqa_start = time.time()

        if settings.is_rag_enabled() and settings.is_laqa_enabled():

            laqa_output = process_query(
                user_query
            )

        elif settings.is_rag_enabled():

            print(
                "LAQA disabled: using raw user query."
            )

            laqa_output = settings.build_raw_query_payload(
                user_query
            )

        else:

            print(
                "RAG disabled: bypassing LAQA and retrieval."
            )

            laqa_output = settings.build_raw_query_payload(
                user_query
            )

        laqa_time = round(
            time.time() - laqa_start,
            2
        )

        # =====================================================
        # 🔹 STEP 2: AGENTIC RAG
        # =====================================================
        print(
            "🔹 Step 2: Agent Execution "
            "(RAG + Retry Loop)..."
        )

        rag_start = time.time()

        agent_result = agent_decision(
            laqa_output
        )

        rag_time = round(
            time.time() - rag_start,
            2
        )

        answer = agent_result.get(
            "answer",
            ""
        )

        docs = agent_result.get(
            "docs",
            []
        )

        doc_ids = agent_result.get(
            "doc_ids",
            []
        )

        eval_result = agent_result.get(
            "eval",
            {}
        )

        # =====================================================
        # 🔹 STEP 3: XAI
        # =====================================================
        print(
            "🔹 Step 3: Explainability Layer..."
        )

        xai_start = time.time()

        explanation = generate_explanation(

            answer,

            docs,

            eval_result,

            user_query
        )

        xai_time = round(
            time.time() - xai_start,
            2
        )

        # =====================================================
        # 🔹 CONFIDENCE
        # =====================================================
        confidence = explanation.get(
            "confidence",
            0.65
        )

        if (
            confidence is None
            or confidence <= 0
        ):

            confidence = 0.65

        # =====================================================
        # 🔹 TOTAL TIME
        # =====================================================
        total_time = round(
            time.time() - start_time,
            2
        )

        # =====================================================
        # 🔹 PIPELINE METRICS
        # =====================================================
        metrics = {

            "laqa_time": laqa_time,

            "rag_time": rag_time,

            "xai_time": xai_time,

            "total_time": total_time,

            "retrieval_score": eval_result.get(
                "retrieval_score",
                0
            ),

            "grounding_score": eval_result.get(
                "grounding_score",
                0
            ),

            "hallucination_risk": eval_result.get(
                "hallucination_risk",
                "medium"
            ),

            "answer_relevance": eval_result.get(
                "answer_relevance",
                0
            )
        }

        # =====================================================
        # 🔹 FINAL OUTPUT
        # =====================================================
        return {

            "answer": answer,

            "confidence": confidence,

            # 🔥 evaluation
            "sources": doc_ids,

            # 🔥 UI
            "source_texts": docs,

            "context_docs": agent_result.get(
                "context_docs",
                docs
            ),

            "candidate_texts": agent_result.get(
                "candidate_texts",
                []
            ),

            # 🔥 XAI
            "explanation": explanation,

            # 🔥 evaluation/debug
            "evaluation": eval_result,

            # 🔥 diagnostics
            "metrics": metrics,

            # 🔥 LAQA visibility
            "query_analysis": {

                "laqa_enabled": settings.is_laqa_enabled(),

                "intent": laqa_output.get(
                    "intent"
                ),

                "query_type": laqa_output.get(
                    "query_type"
                ),

                "expanded_query": laqa_output.get(
                    "expanded_query"
                ),

                "keywords": laqa_output.get(
                    "keywords"
                ),

                "query_metadata": laqa_output.get(
                    "query_metadata",
                    {}
                )
            }
        }

    # =========================================================
    # 🔹 GLOBAL FAILURE HANDLER
    # =========================================================
    except Exception as e:

        print("\n❌ PIPELINE FAILURE")
        print(traceback.format_exc())

        return {

            "answer": (
                "The oncology AI pipeline encountered "
                "an internal processing error."
            ),

            "confidence": 0.1,

            "sources": [],

            "source_texts": [],

            "context_docs": [],

            "explanation": {

                "supporting_sentences": [],

                "reasoning": (
                    "Pipeline execution failed."
                ),

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

                "total_time": round(
                    time.time() - start_time,
                    2
                )
            },

            "query_analysis": {},

            "error": str(e)
        }


# =========================================================
# 🔹 CLI MODE
# =========================================================
if __name__ == "__main__":
    
    print("\n🧠 Oncology AI Assistant")
    print("Type 'exit' to quit\n")

    while True:

        user_query = input(
            "Enter your medical query: "
        ).strip()

        if user_query.lower() == "exit":
            break

        result = handle_query(
            user_query
        )

        # =====================================================
        # 🔹 OUTPUT
        # =====================================================
        print("\n=== FINAL OUTPUT ===")

        print("\n🩺 Answer:\n")
        print(result["answer"])

        print(
            "\n📊 Confidence:",
            result["confidence"]
        )

        # =====================================================
        # 🔹 SUPPORTING EVIDENCE
        # =====================================================
        print("\n📄 Supporting Sentences:")

        supporting = result[
            "explanation"
        ].get(
            "supporting_sentences",
            []
        )

        if supporting:

            for s in supporting:

                print("-", s)

        else:

            print("- No strong supporting evidence")

        # =====================================================
        # 🔹 REASONING
        # =====================================================
        print("\n🧠 Reasoning:")

        print(
            result["explanation"].get(
                "reasoning",
                ""
            )
        )

        # =====================================================
        # 🔹 QUERY ANALYSIS
        # =====================================================
        qa = result.get(
            "query_analysis",
            {}
        )

        print("\n🔍 Query Analysis:")

        print(
            "Intent:",
            qa.get("intent")
        )

        print(
            "Query Type:",
            qa.get("query_type")
        )

        print(
            "Expanded Query:",
            qa.get("expanded_query")
        )

        # =====================================================
        # 🔹 METRICS
        # =====================================================
        metrics = result.get(
            "metrics",
            {}
        )

        print("\n⚙️ Metrics:")

        for k, v in metrics.items():

            print(f"{k}: {v}")

        # =====================================================
        # 🔹 SOURCES
        # =====================================================
        print(
            "\n📚 Source IDs:",
            result["sources"]
        )

        print("\n" + "=" * 60)
