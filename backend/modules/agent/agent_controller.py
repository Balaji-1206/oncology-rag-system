import settings
from modules.retrieval.hybrid_retriever import hybrid_search
from modules.generator.medgemma import generate_answer
from modules.agent.evaluator import evaluate_answer
from modules.agent.strategy import choose_strategy
from modules.agent.memory import AgentMemory
from modules.agent.semantic_cache import lookup_semantic_cache, add_to_semantic_cache


def safe_expand_query(query: str, intent: str, query_type: str, retry_level: int = 1) -> str:
    """Enriches query based on intent, category, and retry attempt level."""
    if query_type == "definition":
        extras = [
            "definition disease overview",
            "medical explanation pathology",
            "clinical characteristics oncology"
        ]
    elif query_type == "list":
        extras = [
            "oncology classification categories",
            "major cancer types classification",
            "common malignant neoplasm categories"
        ]
    elif query_type == "ranking":
        extras = [
            "ranked oncology categories",
            "most common cancer ranking",
            "clinical prevalence importance"
        ]
    elif query_type == "symptoms":
        extras = [
            "symptoms clinical signs presentation",
            "early manifestations oncology",
            "clinical findings disease presentation"
        ]
    elif query_type == "treatment":
        extras = [
            "treatment therapy management",
            "standard oncology treatment",
            "clinical management options"
        ]
    elif intent == "comparison":
        extras = [
            "comparison differences effectiveness",
            "clinical distinction oncology",
            "comparative medical features"
        ]
    elif query_type == "epidemiology":
        extras = [
            "incidence prevalence statistics",
            "frequency percentage occurrence",
            "clinical statistics epidemiology"
        ]
    elif query_type == "yesno":
        extras = [
            "medical oncology evidence",
            "clinical evidence support",
            "oncology medical findings"
        ]
    elif intent == "exploratory":
        extras = [
            "oncology diagnosis treatment",
            "clinical oncology overview",
            "oncology evidence summary"
        ]
    else:
        extras = [
            "medical information oncology",
            "clinical overview oncology",
            "oncology evidence summary"
        ]

    idx = min(retry_level - 1, len(extras) - 1)
    extra_text = extras[idx]

    words = (query + " " + extra_text).split()
    unique = []
    seen = set()
    for w in words:
        if w.lower() not in seen:
            unique.append(w)
            seen.add(w.lower())

    return " ".join(unique)[:300]


def is_generic_answer(answer: str) -> bool:
    if not answer:
        return True

    text = answer.lower().strip()
    generic_patterns = [
        "cannot answer",
        "insufficient context",
        "no information provided",
        "context does not state",
        "unable to determine",
        "not mentioned in the provided",
        "no specific details are given"
    ]

    for p in generic_patterns:
        if p in text:
            return True

    return len(text.split()) < 12


def weak_answer(answer: str) -> bool:
    if is_generic_answer(answer):
        return True
    return len(answer.strip().split()) < 15


def safe_fallback_answer() -> str:
    return (
        "Based on available oncology literature, "
        "a specific grounded answer could not be fully synthesized. "
        "Please refine the query or consult official clinical guidelines."
    )


def compress_context(docs: list, max_length: int = 1500) -> list:
    """Limits total character length of context sent to LLM."""
    compressed = []
    total_len = 0

    for doc in docs:
        if total_len + len(doc) > max_length:
            remaining = max_length - total_len
            if remaining > 200:
                compressed.append(doc[:remaining] + "...")
            break
        compressed.append(doc)
        total_len += len(doc)

    return compressed


def evolve_retry_strategy(laqa_output: dict, action: str, attempt: int) -> dict:
    """Adjusts retrieval parameters (k, query expansion) for retry iterations."""
    new_output = dict(laqa_output)

    if action == "increase_k":
        current_k = new_output.get("retrieval_k", 5)
        new_output["retrieval_k"] = min(current_k + 4, 15)

    elif action == "expand_query":
        base_query = new_output.get("expanded_query") or new_output.get("original_query", "")
        intent = new_output.get("intent", "factual")
        query_type = new_output.get("query_type", "general")

        new_output["expanded_query"] = safe_expand_query(
            base_query,
            intent,
            query_type,
            retry_level=attempt
        )
        new_output["expansion_source"] = "retry_heuristic"

    elif action == "fallback_broad_search":
        original_query = new_output.get("original_query", "")
        new_output["expanded_query"] = original_query + " oncology overview clinical guidelines"
        new_output["retrieval_k"] = 12

    return new_output


def agent_decision(laqa_output: dict) -> dict:
    """Orchestrates agentic retry loop: retrieval -> generation -> evaluation -> retry/accept."""
    raw_query_str = laqa_output.get("original_query") or laqa_output.get("expanded_query", "")

    # Check Semantic Cache
    cached_response = lookup_semantic_cache(raw_query_str)
    if cached_response is not None:
        print("  ⚡ [CACHE HIT] Returning cached response")
        return cached_response

    best_result = None
    best_score = -1.0
    memory = AgentMemory()
    max_attempts = 3

    for attempt in range(1, max_attempts + 1):
        print(f"\n  🤖 Loop Iteration {attempt}/{max_attempts}")

        # Retrieval
        retrieval_result = hybrid_search(laqa_output)
        docs = retrieval_result.get("texts", [])
        doc_ids = retrieval_result.get("ids", [])
        retrieval_score = retrieval_result.get("retrieval_score", 0.0)
        reranker_confidence = retrieval_result.get("reranker_confidence", 0.0)
        query_metadata = retrieval_result.get("query_metadata", {})

        # Low retrieval score check
        if retrieval_score < 0.25 and attempt < max_attempts:
            print("  ⚠️ Low retrieval score, triggering fallback search")
            laqa_output = evolve_retry_strategy(laqa_output, "fallback_broad_search", attempt)
            continue

        compressed_docs = compress_context(docs)

        # Answer Generation
        answer = generate_answer(
            query=raw_query_str,
            docs=compressed_docs,
            intent=laqa_output.get("intent", "factual"),
            query_type=laqa_output.get("query_type", "general"),
            query_metadata=query_metadata,
            keywords=laqa_output.get("keywords", [])
        )

        # Evaluation
        eval_result = evaluate_answer(
            query=raw_query_str,
            docs=compressed_docs,
            answer=answer,
            retrieval_score=retrieval_score,
            intent=laqa_output.get("intent", "factual"),
            query_type=laqa_output.get("query_type", "general")
        )

        # Update Memory
        memory.add_step({
            "attempt": attempt,
            "expanded_query": laqa_output.get("expanded_query", ""),
            "score": eval_result.get("score"),
            "confidence": eval_result.get("confidence"),
            "retrieval_score": retrieval_score,
            "reranker_confidence": reranker_confidence,
            "missing_information": eval_result.get("missing_information"),
            "grounding_score": eval_result.get("grounding_score"),
            "answer": answer[:250]
        })

        # Track Best Result
        eval_score_norm = float(eval_result.get("score", 0)) / 10.0
        current_score = (0.45 * eval_score_norm) + (0.30 * retrieval_score) + (0.25 * reranker_confidence)
        if weak_answer(answer):
            current_score *= 0.75

        if current_score > best_score:
            best_score = current_score
            best_result = {
                "answer": answer,
                "docs": docs,
                "context_docs": compressed_docs,
                "doc_ids": doc_ids,
                "eval": eval_result,
                "candidate_texts": retrieval_result.get("candidate_texts", [])
            }

        # Strategy Choice
        action = choose_strategy(eval_result, attempt)
        print(f"  🎯 Strategy : {action}")

        # Memory Overrides
        if memory.repeated_failure() and action == "expand_query":
            print("  🔁 Memory: repeated failure → escalating to increase_k")
            action = "increase_k"

        if memory.query_drift_detected():
            print("  ⚠️ Memory: query drift detected → returning best result early")
            if best_result is not None:
                return best_result

        # Accept Check
        if action == "accept":
            if weak_answer(answer):
                print("  ⚠️ Prevented weak accept, continuing retry")
            else:
                add_to_semantic_cache(raw_query_str, best_result)
                return best_result

        # Evolve for next attempt
        laqa_output = evolve_retry_strategy(laqa_output, action, attempt)

    print("\n  ⚠️ Max attempts reached, returning best available result")
    if best_result is not None:
        add_to_semantic_cache(raw_query_str, best_result)
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
        },
        "candidate_texts": []
    }
