from modules.retrieval.hybrid_retriever import hybrid_search
from modules.generator.medgemma import generate_answer
from modules.agent.evaluator import evaluate_answer
from modules.agent.strategy import choose_strategy
from modules.agent.memory import AgentMemory
import settings


# =========================================================
# 🔹 SAFE QUERY EXPANSION
# =========================================================
def safe_expand_query(
    query,
    intent,
    query_type,
    retry_level=1
):

    # =====================================================
    # 🔹 DEFINITION
    # =====================================================
    if query_type == "definition":

        extras = [

            "definition disease overview",

            "medical explanation pathology",

            "clinical characteristics oncology"
        ]

    # =====================================================
    # 🔹 LIST
    # =====================================================
    elif query_type == "list":

        extras = [

            "oncology classification categories",

            "major cancer types classification",

            "common malignant neoplasm categories"
        ]

    # =====================================================
    # 🔹 RANKING
    # =====================================================
    elif query_type == "ranking":

        extras = [

            "ranked oncology categories",

            "most common cancer ranking",

            "clinical prevalence importance"
        ]

    # =====================================================
    # 🔹 SYMPTOMS
    # =====================================================
    elif query_type == "symptoms":

        extras = [

            "symptoms clinical signs presentation",

            "early manifestations oncology",

            "clinical findings disease presentation"
        ]

    # =====================================================
    # 🔹 TREATMENT
    # =====================================================
    elif query_type == "treatment":

        extras = [

            "treatment therapy management",

            "standard oncology treatment",

            "clinical management options"
        ]

    # =====================================================
    # 🔹 COMPARISON
    # =====================================================
    elif intent == "comparison":

        extras = [

            "comparison differences effectiveness",

            "clinical distinction oncology",

            "comparative medical features"
        ]

    # =====================================================
    # 🔹 YES/NO
    # =====================================================
    elif query_type == "yesno":

        extras = [

            "medical oncology evidence",

            "clinical evidence support",

            "oncology medical findings"
        ]

    # =====================================================
    # 🔹 EXPLORATORY
    # =====================================================
    elif intent == "exploratory":

        extras = [

            "oncology diagnosis treatment",

            "clinical oncology overview",

            "oncology evidence summary"
        ]

    # =====================================================
    # 🔹 GENERAL
    # =====================================================
    else:

        extras = [

            "oncology clinical information",

            "medical oncology evidence",

            "oncology disease information"
        ]

    retry_level = min(
        retry_level,
        len(extras)
    )

    extra = " ".join(
        extras[:retry_level]
    )

    existing = set(
        query.lower().split()
    )

    extra_words = [

        w for w in extra.split()

        if w.lower() not in existing
    ]

    expanded = (
        query
        + " "
        + " ".join(extra_words)
    )

    return expanded[:320]


# =========================================================
# 🔹 RETRIEVAL FAILURE DETECTION
# =========================================================
def retrieval_failed(
    docs,
    retrieval_score,
    reranker_confidence
):

    if not docs:
        return True

    if retrieval_score < 0.32:
        return True

    if (
        retrieval_score < 0.40
        and reranker_confidence < 0.35
    ):
        return True

    return False


# =========================================================
# 🔹 EDUCATIONAL QUERY REPAIR
# =========================================================
def educational_query_repair(
    laqa_output
):

    if not settings.is_laqa_enabled():
        return laqa_output

    query_type = laqa_output.get(
        "query_type",
        "general"
    )

    original = laqa_output[
        "original_query"
    ]

    if query_type == "definition":

        laqa_output["expanded_query"] = (
            original
            +
            " definition disease overview pathology"
        )

    elif query_type == "symptoms":

        laqa_output["expanded_query"] = (
            original
            +
            " symptoms clinical signs presentation"
        )

    elif query_type == "yesno":

        laqa_output["expanded_query"] = (
            original
            +
            " oncology medical evidence"
        )

    return laqa_output


# =========================================================
# 🔹 CONTEXT COMPRESSION
# =========================================================
def compress_docs(
    docs,
    query_type
):

    compressed = []

    if query_type in [
        "list",
        "ranking"
    ]:

        limit = 850

    else:

        limit = 700

    for d in docs[:4]:

        compressed.append(
            d[:limit]
        )

    return compressed


# =========================================================
# 🔹 WEAK ANSWER DETECTION
# =========================================================
def weak_answer(
    answer
):

    if len(answer.strip()) < 25:
        return True

    weak_patterns = [

        "not enough information",

        "insufficient evidence",

        "unable to determine",

        "context does not provide",

        "the context discusses",

        "based on the context"
    ]

    ans = answer.lower()

    for p in weak_patterns:

        if p in ans:
            return True

    return False


# =========================================================
# 🔹 RETRY EVOLUTION
# =========================================================
def evolve_retry_strategy(
    laqa_output,
    action,
    attempt
):

    if (
        not settings.is_laqa_enabled()
        and action == "expand_query"
    ):

        laqa_output["expanded_query"] = laqa_output.get(
            "original_query",
            laqa_output.get("expanded_query", "")
        )

        return laqa_output

    query_type = laqa_output.get(
        "query_type",
        "general"
    )

    intent = laqa_output.get(
        "intent",
        "factual"
    )

    # =====================================================
    # 🔹 EXPAND QUERY
    # =====================================================
    if action == "expand_query":

        laqa_output[
            "expanded_query"
        ] = safe_expand_query(

            laqa_output[
                "original_query"
            ],

            intent,

            query_type,

            retry_level=attempt + 1
        )

    # =====================================================
    # 🔹 INCREASE K
    # =====================================================
    elif action == "increase_k":

        current_k = laqa_output.get(
            "retrieval_k",
            5
        )

        if attempt == 0:

            new_k = current_k + 2

        else:

            new_k = current_k + 3

        laqa_output[
            "retrieval_k"
        ] = min(new_k, 12)

    return laqa_output


# =========================================================
# 🔹 FINAL FALLBACK
# =========================================================
def safe_fallback_answer():

    return (
        "I could not retrieve sufficiently "
        "reliable oncology evidence to provide "
        "a confident medical answer."
    )


# =========================================================
# 🔹 MAIN AGENT LOOP
# =========================================================
def agent_decision(query_input):

    # =====================================================
    # 🔹 SUPPORT BOTH RAW QUERY + LAQA OUTPUT
    # =====================================================
    if isinstance(query_input, str):

        laqa_output = settings.build_raw_query_payload(
            query_input
        )

    else:

        laqa_output = query_input

    memory = AgentMemory()

    query_type = laqa_output.get(
        "query_type",
        "general"
    )

    # =====================================================
    # 🔹 ATTEMPTS
    # =====================================================
    max_attempts = 3

    # =====================================================
    # 🔹 BEST RESULT
    # =====================================================
    best_result = None

    best_score = -1

    # =====================================================
    # 🔹 LOOP
    # =====================================================
    for attempt in range(max_attempts):

        print(f"\n🔁 ATTEMPT {attempt + 1}")

        # =====================================================
        # 🔹 EDUCATIONAL RESCUE
        # =====================================================
        if (
            settings.is_laqa_enabled()
            and
            attempt == 1
            and query_type in [

                "definition",

                "symptoms",

                "yesno"
            ]
        ):

            print(
                "🩺 Educational query repair"
            )

            laqa_output = educational_query_repair(
                laqa_output
            )

        # =====================================================
        # 🔹 RETRIEVAL
        # =====================================================
        retrieval_result = hybrid_search(
            laqa_output,
            None
        )

        docs = retrieval_result.get(
            "texts",
            []
        )

        doc_ids = retrieval_result.get(
            "ids",
            []
        )

        retrieval_score = retrieval_result.get(
            "retrieval_score",
            0.4
        )

        reranker_confidence = retrieval_result.get(
            "reranker_confidence",
            0.0
        )

        # =====================================================
        # 🔹 RETRIEVAL FAILURE
        # =====================================================
        if retrieval_failed(
            docs,
            retrieval_score,
            reranker_confidence
        ):

            print(
                "⚠️ Retrieval quality too low"
            )

            if attempt < max_attempts - 1:

                laqa_output = evolve_retry_strategy(
                    laqa_output,
                    "increase_k",
                    attempt
                )

                continue

            return {

                "answer": safe_fallback_answer(),

                "docs": [],

                "context_docs": [],

                "doc_ids": [],

                "eval": {

                    "score": 2,

                    "confidence": 0.25,

                    "needs_retry": False,

                    "retrieval_score": retrieval_score,

                    "reranker_confidence": reranker_confidence
                }
            }

        # =====================================================
        # 🔹 CONTEXT
        # =====================================================
        compressed_docs = compress_docs(
            docs,
            query_type
        )

        context = "\n".join(
            compressed_docs[:3]
        )[:2600]

        # =====================================================
        # 🔹 GENERATION
        # =====================================================
        agent_input = {

            "query": laqa_output,

            "context": compressed_docs
        }

        answer = generate_answer(
            agent_input
        )

        # =====================================================
        # 🔹 WEAK ANSWER
        # =====================================================
        if weak_answer(answer):

            print(
                "⚠️ Weak answer detected"
            )

        # =====================================================
        # 🔹 EVALUATION
        # =====================================================
        print(
            "EVALUATOR RECEIVED DOCS:",
            len(compressed_docs)
        )

        eval_result = evaluate_answer(
            laqa_output["expanded_query"],
            context,
            answer
        )

        eval_result[
            "retrieval_score"
        ] = retrieval_score

        eval_result[
            "reranker_confidence"
        ] = reranker_confidence

        eval_result[
            "retrieval_diagnostics"
        ] = retrieval_result.get(
            "retrieval_diagnostics",
            []
        )


        # =====================================================
        # 🔹 MEMORY
        # =====================================================
        memory.add({

            "attempt": attempt + 1,

            "query": laqa_output[
                "expanded_query"
            ],

            "score": eval_result.get(
                "score"
            ),

            "confidence": eval_result.get(
                "confidence"
            ),

            "retrieval_score": retrieval_score,

            "reranker_confidence": reranker_confidence,

            "missing_information": eval_result.get(
                "missing_information"
            ),

            "grounding_score": eval_result.get(
                "grounding_score"
            ),

            "answer": answer[:250]
        })

        # =====================================================
        # 🔹 BEST RESULT TRACKING
        # =====================================================
        current_score = (

            0.45 * eval_result.get(
                "score",
                0
            )

            +

            0.30 * retrieval_score

            +

            0.25 * reranker_confidence
        )

        if weak_answer(answer):

            current_score *= 0.75

        if current_score > best_score:

            best_score = current_score

            best_result = {

                "answer": answer,

                "docs": docs,

                "context_docs": compressed_docs,

                "doc_ids": doc_ids,

                "eval": eval_result
            }

        # =====================================================
        # 🔹 STRATEGY
        # =====================================================
        action = choose_strategy(
            eval_result,
            attempt
        )

        print("ACTION:", action)

        # =====================================================
        # 🔹 ACCEPT
        # =====================================================
        if action == "accept":

            if weak_answer(answer):

                print(
                    "⚠️ Prevented weak accept"
                )

            else:

                return best_result

        # =====================================================
        # 🔹 RETRY EVOLUTION
        # =====================================================
        laqa_output = evolve_retry_strategy(
            laqa_output,
            action,
            attempt
        )

    # =====================================================
    # 🔹 MAX ATTEMPTS
    # =====================================================
    print("\n⚠️ Max attempts reached")

    if best_result is not None:

        return best_result

    return {

        "answer": safe_fallback_answer(),

        "docs": [],

        "context_docs": [],

        "doc_ids": [],

        "eval": {

            "score": 2,

            "confidence": 0.25,

            "needs_retry": False,

            "retrieval_score": 0.0,

            "reranker_confidence": 0.0
        }
    }
