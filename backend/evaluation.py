import json
import numpy as np
import io
import contextlib
from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm

import settings
from app import handle_query

from metrics import *


# =========================================================
# 🔹 LOAD DATASET
# =========================================================
def load_dataset(path):

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


# =========================================================
# 🔹 SAFE AVG
# =========================================================
def avg(results, key):

    vals = [

        r.get(key, 0)

        for r in results
    ]

    if not vals:
        return 0

    return float(np.mean(vals))


# =========================================================
# 🔹 QUERY TYPE
# =========================================================
def detect_query_type(q):

    q = q.lower()

    if "what is" in q:
        return "definition"

    if "symptom" in q:
        return "symptoms"

    if q.startswith(
        ("can", "does", "is", "are")
    ):
        return "yesno"

    if (
        "compare" in q
        or "difference" in q
    ):
        return "comparison"

    return "general"


# =========================================================
# 🔹 REFUSAL
# =========================================================
def refusal_detected(answer):

    patterns = [

        "not enough information",

        "insufficient information",

        "cannot determine",
    ]

    answer = answer.lower()

    return any(
        p in answer
        for p in patterns
    )


# =========================================================
# 🔹 MAIN EVALUATION
# =========================================================
def evaluate(dataset_path):

    data = load_dataset(
        dataset_path
    )

    results = []

    predictions = []

    references = []

    print("\n🚀 Running Evaluation...\n")

    total_questions = min(
        10,
        len(data)
    )

    progress_bar = tqdm(
        data[:10],
        desc="🧪 Evaluating",
        ncols=120
    )

    for idx, item in enumerate(
        progress_bar,
        start=1
    ):

        # =====================================================
        # 🔹 QUESTION HEADER
        # =====================================================
        q = item["q"]

        print(
            f"\n{'='*80}"
        )

        print(
            f"🧪 Q{idx}/{total_questions}"
        )

        print(
            f"❓ {q}"
        )

        print(
            f"{'='*80}\n"
        )

        gt = item["a"]

        query_type = detect_query_type(
            q
        )

        # =====================================================
        # 🔹 PIPELINE
        # =====================================================
        with contextlib.redirect_stdout(
            io.StringIO()
        ):

            result = handle_query(q)

        pred = result.get(
            "answer",
            ""
        )

        pred_ids = result.get(
            "sources",
            []
        )

        contexts = result.get(
            "source_texts",
            []
        )

        context_docs = result.get(
            "context_docs",
            []
        ) or contexts

        print(
            "EVALUATOR RECEIVED DOCS:",
            len(context_docs)
        )

        confidence = result.get(
            "confidence",
            0.5
        )

        explanation = result.get(
            "explanation",
            {}
        )

        evaluation = result.get(
            "evaluation",
            {}
        )

        # =====================================================
        # 🔹 GENERATION METRICS
        # =====================================================
        prepare_metric_embeddings(
            q,
            gt,
            pred,
            context_docs
        )

        gt_ids = pred_ids[:1] if pred_ids else []

        print(
            "FAITHFULNESS DOC COUNT:",
            len(context_docs)
        )

        print(
            "CONTEXT RELEVANCY DOC COUNT:",
            len(context_docs)
        )

        with ThreadPoolExecutor(
            max_workers=8
        ) as executor:

            futures = {

                "bleu": executor.submit(
                    compute_bleu_scores,
                    gt,
                    pred
                ),

                "gleu": executor.submit(
                    compute_gleu_score,
                    gt,
                    pred
                ),

                "meteor": executor.submit(
                    compute_meteor_score,
                    gt,
                    pred
                ),

                "distinct": executor.submit(
                    compute_distinct_scores,
                    pred
                ),

                "accuracy_f1": executor.submit(
                    compute_accuracy_f1,
                    gt,
                    pred
                ),

                "rouge": executor.submit(
                    compute_rouge_scores,
                    gt,
                    pred
                ),

                "sbert_similarity": executor.submit(
                    compute_sbert_similarity,
                    gt,
                    pred
                ),

                "llm_metrics": executor.submit(
                    combined_llm_evaluation,
                    q,
                    gt,
                    pred
                ),

                "faithfulness": executor.submit(
                    compute_faithfulness,
                    pred,
                    context_docs
                ),

                "context_rel": executor.submit(
                    context_relevance,
                    context_docs,
                    q
                ),

                "answer_rel": executor.submit(
                    answer_relevance,
                    pred,
                    q
                ),

                "precision": executor.submit(
                    precision_at_k,
                    pred_ids,
                    gt_ids
                ),

                "recall": executor.submit(
                    recall_at_k,
                    pred_ids,
                    gt_ids
                ),

                "mrr": executor.submit(
                    mrr,
                    pred_ids,
                    gt_ids
                ),

                "ndcg": executor.submit(
                    ndcg,
                    pred_ids,
                    gt_ids
                ),

                "hit_rate": executor.submit(
                    hit_rate,
                    pred_ids,
                    gt_ids
                )
            }

            bleu = futures["bleu"].result()
            gleu = futures["gleu"].result()
            meteor = futures["meteor"].result()
            distinct = futures["distinct"].result()
            accuracy_f1 = futures["accuracy_f1"].result()
            rouge = futures["rouge"].result()

            sbert_similarity = {

                "sbert_similarity": futures[
                    "sbert_similarity"
                ].result()
            }

            llm_all = futures["llm_metrics"].result()

            llm_judge = {

                "llm_judge_score": llm_all.get(
                    "llm_judge_score",
                    0.0
                ),

                "llm_judge_reason": llm_all.get(
                    "llm_judge_reason",
                    "judge_failed"
                )
            }

            scope = {

                "scope_safety": llm_all.get(
                    "scope_safety",
                    0.0
                ),

                "scope_completeness": llm_all.get(
                    "scope_completeness",
                    0.0
                ),

                "scope_originality": llm_all.get(
                    "scope_originality",
                    0.0
                ),

                "scope_precision": llm_all.get(
                    "scope_precision",
                    0.0
                ),

                "scope_efficiency": llm_all.get(
                    "scope_efficiency",
                    0.0
                ),

                "scope_weighted_total": llm_all.get(
                    "scope_weighted_total",
                    0.0
                )
            }

            faithfulness = futures["faithfulness"].result()

            context_rel = futures["context_rel"].result()
            answer_rel = futures["answer_rel"].result()

        # =====================================================
        # 🔹 RETRIEVAL
        # =====================================================
        retrieval = {

            "precision": futures["precision"].result(),

            "recall": futures["recall"].result(),

            "mrr": futures["mrr"].result(),

            "ndcg": futures["ndcg"].result(),

            "hit_rate": futures["hit_rate"].result()
        }

        # =====================================================
        # 🔹 MEDICAL RAG METRICS
        # =====================================================
        medical_metrics = {

            "faithfulness": faithfulness,

            "context_rel": context_rel,

            "answer_rel": answer_rel,

            "confidence": confidence,

            "grounding_score": evaluation.get(
                "grounding_score",
                0
            ),

            "retrieval_score": evaluation.get(
                "retrieval_score",
                0
            ),

            "hallucination_low": int(
                evaluation.get(
                    "hallucination_risk",
                    "medium"
                ) == "low"
            ),

            "safe_refusal": int(
                refusal_detected(pred)
            ),

            "grounded_answer": int(
                explanation.get(
                    "grounded",
                    False
                )
            ),

            "answer_quality": evaluation.get(
                "score",
                0
            ),

            "agent_iters": evaluation.get(
                "agent_iterations",
                1
            ),

            "rerank_score": evaluation.get(
                "reranker_confidence",
                0
            ),

            "query_type": query_type
        }

        # =====================================================
        # 🔹 FINAL RESULT
        # =====================================================
        res = {

            **bleu,

            **gleu,

            **meteor,

            **scope,

            **distinct,

            **accuracy_f1,

            **rouge,

            **sbert_similarity,

            **llm_judge,

            **retrieval,

            **medical_metrics
        }

        results.append(res)

        predictions.append(pred)

        references.append(gt)

        # =====================================================
        # 🔹 STRUCTURED SUMMARY
        # =====================================================
        print(
            f"✅ Completed Q{idx}"
        )

        print(
            f"📊 Grounding : "
            f"{evaluation.get('grounding_score',0):.2f}"
        )

        print(
            f"📊 Retrieval : "
            f"{evaluation.get('retrieval_score',0):.2f}"
        )

        print(
            f"📊 Confidence: "
            f"{confidence:.2f}"
        )

        print(
            f"📊 Quality   : "
            f"{evaluation.get('score',0)}"
        )

    # =========================================================
    # 🔹 BERT SCORE
    # =========================================================
    bert = compute_bertscore(
        references,
        predictions
    )

    return results, bert


# =========================================================
# 🔹 REPORT
# =========================================================
def print_report(results, bert):

    print("\n" + "=" * 80)

    print(
        "ONCOLOGY RAG - COMPLETE EVALUATION REPORT"
    )

    print(
        "Agentic Oncology RAG"
    )

    print(
        f"LAQA enabled        : {settings.is_laqa_enabled()}"
    )

    print(
        f"MRL enabled         : {settings.is_mrl_enabled()}"
    )

    print("=" * 80)

    # =====================================================
    # 🔹 BASIC
    # =====================================================
    print(
        f"\nQuestions evaluated : {len(results)}"
    )

    print(
        f"Avg confidence     : "
        f"{avg(results,'confidence'):.4f}"
    )

    print(
        f"Avg agent iters   : "
        f"{avg(results,'agent_iters'):.2f}"
    )

    print(
        f"Avg rerank score  : "
        f"{avg(results,'rerank_score'):.4f}"
    )

    # =====================================================
    # 🔹 RETRIEVAL
    # =====================================================
    print(
        "\n-- Retrieval Quality (k=5) "
        "---------------------------------------------"
    )

    print(
        f"Precision@5        : "
        f"{avg(results,'precision'):.4f}"
    )

    print(
        f"Recall@5           : "
        f"{avg(results,'recall'):.4f}"
    )

    print(
        f"MRR                : "
        f"{avg(results,'mrr'):.4f}"
    )

    print(
        f"NDCG@5             : "
        f"{avg(results,'ndcg'):.4f}"
    )

    print(
        f"Hit-Rate@5         : "
        f"{avg(results,'hit_rate'):.4f}"
    )

    # =====================================================
    # 🔹 GENERATION
    # =====================================================
    print(
        "\n-- Generation Lexical "
        "-----------------------------------------------"
    )

    print(
        f"BLEU-1             : "
        f"{avg(results,'bleu1'):.4f}"
    )

    print(
        f"BLEU-2             : "
        f"{avg(results,'bleu2'):.4f}"
    )

    print(
        f"BLEU-4             : "
        f"{avg(results,'bleu4'):.4f}"
    )

    print(
        f"GLEU               : "
        f"{avg(results,'gleu'):.4f}"
    )

    print(
        f"METEOR             : "
        f"{avg(results,'meteor'):.4f}"
    )

    print(
        f"Accuracy           : "
        f"{avg(results,'accuracy'):.4f}"
    )

    print(
        f"F1                 : "
        f"{avg(results,'f1'):.4f}"
    )

    print(
        f"ROUGE-1            : "
        f"{avg(results,'rouge1'):.4f}"
    )

    print(
        f"ROUGE-2            : "
        f"{avg(results,'rouge2'):.4f}"
    )

    print(
        f"ROUGE-L            : "
        f"{avg(results,'rougeL'):.4f}"
    )

    print(
        f"ROUGE-Lsum         : "
        f"{avg(results,'rougeLsum'):.4f}"
    )

    # =====================================================
    # 🔹 SEMANTIC
    # =====================================================
    print(
        "\n-- Generation Semantic "
        "----------------------------------------------"
    )

    print(
        f"BERTScore F1       : "
        f"{bert:.4f}"
    )

    print(
        f"SBERT similarity   : "
        f"{avg(results,'sbert_similarity'):.4f}"
    )

    print(
        f"DISTINCT-1         : "
        f"{avg(results,'distinct1'):.4f}"
    )

    print(
        f"DISTINCT-2         : "
        f"{avg(results,'distinct2'):.4f}"
    )

    # =====================================================
    # 🔹 MEDICAL RAG
    # =====================================================
    print(
        "\n-- Faithfulness & Safety "
        "-----------------------------------------"
    )

    print(
        f"Faithfulness       : "
        f"{avg(results,'faithfulness'):.4f}"
    )

    print(
        f"Context Relevancy  : "
        f"{avg(results,'context_rel'):.4f}"
    )

    print(
        f"Answer relevancy   : "
        f"{avg(results,'answer_rel'):.4f}"
    )

    print(
        f"Grounding Score    : "
        f"{avg(results,'grounding_score'):.4f}"
    )

    print(
        f"Retrieval Score    : "
        f"{avg(results,'retrieval_score'):.4f}"
    )

    print(
        f"Hallucination Safe : "
        f"{avg(results,'hallucination_low'):.4f}"
    )

    print(
        f"Grounded Answers   : "
        f"{avg(results,'grounded_answer'):.4f}"
    )

    print(
        f"Safe Refusal Rate  : "
        f"{avg(results,'safe_refusal'):.4f}"
    )

    print(
        f"Answer Quality     : "
        f"{avg(results,'answer_quality'):.4f}"
    )

    print(
        f"LLM Judge Score    : "
        f"{avg(results,'llm_judge_score'):.4f}"
    )

    # =====================================================
    # 🔹 SCOPE
    # =====================================================
    print(
        "\n-- S.C.O.P.E LLM-as-judge (/5.0) "
        "------------------------------------------------"
    )

    print(
        f"S  Safety         : "
        f"{avg(results,'scope_safety'):.2f}"
    )

    print(
        f"C  Completeness   : "
        f"{avg(results,'scope_completeness'):.2f}"
    )

    print(
        f"O  Originality    : "
        f"{avg(results,'scope_originality'):.2f}"
    )

    print(
        f"P  Precision      : "
        f"{avg(results,'scope_precision'):.2f}"
    )

    print(
        f"E  Efficiency     : "
        f"{avg(results,'scope_efficiency'):.2f}"
    )

    weighted = avg(
        results,
        "scope_weighted_total"
    )

    scores = [

        r.get(
            "scope_weighted_total",
            0
        )

        for r in results
    ]

    std = np.std(scores)

    print(
        f"Weighted Total    : "
        f"{weighted:.2f}/5.00 "
        f"(std={std:.2f})"
    )

    print("=" * 80)


# =========================================================
# 🔹 RUN
# =========================================================
if __name__ == "__main__":

    dataset_path = (
        "backend/complex50.json"
    )

    results, bert = evaluate(
        dataset_path
    )

    print_report(
        results,
        bert
    )
