import json

from profile_pipeline import (
    PipelineProfiler,
    profile_pipeline
)

# =========================================================
# 🔹 IMPORT REAL MODULES
# =========================================================

from modules.laqa.laqa import process_query

from modules.retrieval.hybrid_retriever import (
    hybrid_search
)

from modules.retrieval.reranker import (
    rerank
)

from modules.generator.medgemma import (
    generate_answer
)

from modules.agent.evaluator import (
    evaluate_answer
)

from modules.xai.explain import (
    generate_explanation
)

from metrics import (
    compute_bleu_scores,
    compute_rouge_scores,
    compute_meteor_score,
    compute_sbert_similarity,
    compute_bertscore,
    compute_faithfulness,
    context_relevance,
    answer_relevance
)

# =========================================================
# 🔹 QUESTIONS
# =========================================================

questions = [
    "What are the common side effects of chemotherapy?"
    "In lung cancer staging, what is the significance of a mediastinal lymph node measuring 1.5 cm on a CT scan?",
    "What percentage of patients with Small Cell Lung Cancer (SCLC) have neurologically asymptomatic brain metastases?"
]

# =========================================================
# 🔹 QUESTION RUNNER
# =========================================================

def question_runner(
    question,
    profiler,
    question_id
):

    # -----------------------------------------------------
    # 1️⃣ LAQA
    # -----------------------------------------------------
    with profiler.profile_stage("LAQA"):

        laqa_result = process_query(
            question
        )

        expanded_query = laqa_result.get(
            "expanded_query",
            question
        )

        print("\n🧠 LAQA PARSED:")

        print(
            json.dumps(
                laqa_result,
                indent=2
            )
        )

    # -----------------------------------------------------
    # 2️⃣ QUERY EXPANSION
    # -----------------------------------------------------
    with profiler.profile_stage("Query Expansion"):

        final_query = expanded_query

    # -----------------------------------------------------
    # 3️⃣ DENSE EMBEDDING
    # -----------------------------------------------------
    with profiler.profile_stage("Dense Embedding"):

        profiler.note_embedding_call(
            final_query
        )

    # -----------------------------------------------------
    # 4️⃣ FAISS SEARCH
    # -----------------------------------------------------
    with profiler.profile_stage("FAISS Search"):

        retrieval_results = hybrid_search(
            laqa_result,
            None
        )

    # -----------------------------------------------------
    # 5️⃣ BM25
    # already inside hybrid_search
    # -----------------------------------------------------
    with profiler.profile_stage("BM25 Retrieval"):

        pass

    # -----------------------------------------------------
    # 6️⃣ HYBRID FUSION
    # already inside hybrid_search
    # -----------------------------------------------------
    with profiler.profile_stage("Hybrid Fusion"):

        pass

    # -----------------------------------------------------
    # 🔹 DOCS
    # -----------------------------------------------------
    docs = retrieval_results.get(
        "texts",
        []
    )

    print("\n📄 Top Retrieved Docs:")

    for idx, doc in enumerate(
        docs[:4],
        start=1
    ):

        print(
            f"{idx}. {doc[:250]} ..."
        )

    # -----------------------------------------------------
    # 7️⃣ RERANKER
    # -----------------------------------------------------
    with profiler.profile_stage("Reranker"):

        reranked_docs = rerank(
            final_query,
            docs,
            top_k=5
        )

    # -----------------------------------------------------
    # 8️⃣ CONTEXT BUILDING
    # -----------------------------------------------------
    with profiler.profile_stage("Context Building"):

        context = "\n\n".join(
            reranked_docs
        )

    # -----------------------------------------------------
    # 9️⃣ GENERATION
    # -----------------------------------------------------
    with profiler.profile_stage("MedGemma Generation"):

        profiler.note_ollama_call(
            question
        )

        answer = generate_answer({

            "query": laqa_result,

            "context": reranked_docs
        })

        print("\n🧠 RAW GENERATOR OUTPUT:\n")

        print(answer)

    # -----------------------------------------------------
    # 🔟 EVALUATOR
    # -----------------------------------------------------
    with profiler.profile_stage("Evaluator LLM"):

        profiler.note_evaluator_call(
            answer
        )

        print(
            "EVALUATOR RECEIVED DOCS:",
            len(reranked_docs)
        )

        eval_result = evaluate_answer(
            question,
            context,
            answer
        )

        print("\n🧠 EVAL RAW OUTPUT:\n")

        print(
            json.dumps(
                eval_result,
                indent=2
            )
        )

    # -----------------------------------------------------
    # 1️⃣1️⃣ LEXICAL METRICS
    # -----------------------------------------------------
    with profiler.profile_stage("BLEU/ROUGE/METEOR"):

        reference = context[:1000]

        bleu = compute_bleu_scores(
            reference,
            answer
        )

        rouge = compute_rouge_scores(
            reference,
            answer
        )

        meteor = compute_meteor_score(
            reference,
            answer
        )

    # -----------------------------------------------------
    # 1️⃣2️⃣ SEMANTIC METRICS
    # -----------------------------------------------------
    with profiler.profile_stage("SBERT/BERTScore"):

        sbert = compute_sbert_similarity(
            reference,
            answer
        )

        bertscore = compute_bertscore(
            reference,
            answer
        )

    # -----------------------------------------------------
    # 1️⃣3️⃣ GROUNDING
    # -----------------------------------------------------
    with profiler.profile_stage("Grounding Metric"):

        grounding = eval_result.get(
            "grounding_score",
            0
        )

    # -----------------------------------------------------
    # 1️⃣4️⃣ FAITHFULNESS
    # -----------------------------------------------------
    with profiler.profile_stage("Faithfulness Metric"):

        print(
            "FAITHFULNESS DOC COUNT:",
            len(reranked_docs)
        )

        faithfulness = compute_faithfulness(
            answer,
            reranked_docs
        )

    # -----------------------------------------------------
    # 1️⃣5️⃣ RELEVANCE
    # -----------------------------------------------------
    with profiler.profile_stage("Relevance Metric"):

        print(
            "CONTEXT RELEVANCY DOC COUNT:",
            len(reranked_docs)
        )

        context_rel = context_relevance(
            reranked_docs,
            question
        )

        relevance = answer_relevance(
            answer,
            question
        )

    # -----------------------------------------------------
    # 1️⃣6️⃣ SCOPE
    # -----------------------------------------------------
    with profiler.profile_stage("S.C.O.P.E Evaluation"):

        scope = eval_result.get(
            "score",
            0
        )

    # -----------------------------------------------------
    # 1️⃣7️⃣ XAI
    # -----------------------------------------------------
    with profiler.profile_stage("XAI Layer"):

        explanation = generate_explanation(
            answer,
            reranked_docs,
            eval_result,
            question
        )
    # -----------------------------------------------------
    # 🔹 FINAL
    # -----------------------------------------------------
    print(f"\n✅ Q{question_id} Complete")

    print(
        f"   📊 Quality: {scope:.2f}"
    )

    print(
        f"   📊 Grounding: {grounding:.2f}"
    )

# =========================================================
# 🔹 MAIN
# =========================================================

if __name__ == "__main__":

    print("\n" + "=" * 80)

    print(
        "🚀 Running Enhanced Pipeline Profiling..."
    )

    print("=" * 80)

    profiler = PipelineProfiler()

    records, profiler = profile_pipeline(

        questions,

        question_runner,

        profiler=profiler,

        print_each_question=True,

        print_summary=True
    )

    print("\n" + "=" * 80)

    print(
        "✅ RUN_PROFILE EXECUTION COMPLETE"
    )

    print("=" * 80)
