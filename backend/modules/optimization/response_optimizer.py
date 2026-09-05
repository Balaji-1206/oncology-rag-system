import re


# =========================================================
# 🔹 NOISE & ARTIFACT PATTERNS
# =========================================================
REASONING_MARKERS = [
    "Answer:",
    "ANSWER:",
    "FINAL ANSWER:",
    "Final Answer:",
    "final medical answer:",
    "<unused95>"
]

STRIP_BLOCK_PATTERNS = [
    r"<unused\d+>",
    r"(?s)<think>.*?</think>",
    r"(?s)<analysis>.*?</analysis>",
    r"(?s)<reasoning>.*?</reasoning>",
]

STRIP_LINE_PREFIXES = [
    r"(?im)^(thought|thinking|analysis|reasoning|observation|constraint checklist|step-by-step|scratchpad|chain of thought)\s*:?\s*",
    r"(?im)^(the user wants|the user is asking|system prompt|developer message)\s*:?\s*",
    r"(?im)^(final answer|final medical answer|answer)\s*:?\s*"
]

SEVERE_LEAK_PATTERNS = [
    "system prompt",
    "internal instruction",
    "<think>",
    "<analysis>"
]


# =========================================================
# 🔹 ARTIFACT & NOISE STRIPPING
# =========================================================
def strip_generation_artifacts(raw_text: str) -> tuple[str, int]:
    """Removes thinking tags, prompt echoes, and reasoning prefixes."""
    if not raw_text:
        return "", 0

    text = str(raw_text)
    removed_count = 0

    # 1. Strip leading whole-string markers
    for marker in REASONING_MARKERS:
        if marker in text:
            text = text.split(marker)[-1]
            removed_count += 1

    # 2. Strip multi-line tag blocks (<think>...</think>, etc.)
    for pattern in STRIP_BLOCK_PATTERNS:
        matches = len(re.findall(pattern, text))
        if matches > 0:
            removed_count += matches
            text = re.sub(pattern, "", text)

    # 3. Strip line-by-line prefix noise without deleting valid content
    cleaned_lines = []
    for line in text.split("\n"):
        line_clean = line.strip()
        if not line_clean:
            continue

        # Check if line is purely meta-commentary
        if re.match(r"(?i)^(observation|constraint checklist|reasoning process)\s*:?", line_clean):
            # Check if it has actual medical value or is just scratchpad
            if re.search(r"(?i)\b(fda|indicated|approved|mutation|treatment|therapy|cancer|tumor|stage)\b", line_clean):
                for p in STRIP_LINE_PREFIXES:
                    line_clean = re.sub(p, "", line_clean).strip()
                if line_clean:
                    cleaned_lines.append(line_clean)
            else:
                removed_count += 1
                continue
        else:
            for p in STRIP_LINE_PREFIXES:
                if re.search(p, line_clean):
                    removed_count += 1
                    line_clean = re.sub(p, "", line_clean).strip()

            if line_clean:
                cleaned_lines.append(line_clean)

    return "\n".join(cleaned_lines).strip(), removed_count


# =========================================================
# 🔹 DEDUPLICATION & REDUNDANCY PRUNING
# =========================================================
def prune_redundancy(text: str) -> tuple[str, int]:
    """Filters duplicate bullet points, repeating lines, and excessive whitespace."""
    if not text:
        return "", 0

    lines = []
    seen = set()
    duplicates_removed = 0

    for raw_line in text.split("\n"):
        line = raw_line.strip()
        if not line:
            continue

        # Normalize line by stripping leading markdown bullet points / numbering
        normalized = re.sub(r"^[\s*•\-\d.)]+", "", line).strip().lower()
        normalized = re.sub(r"\s+", " ", normalized)

        key = normalized if normalized else line.lower()
        if key in seen:
            duplicates_removed += 1
            continue

        seen.add(key)
        lines.append(line)

    cleaned = "\n".join(lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)

    return cleaned.strip(), duplicates_removed


# =========================================================
# 🔹 CONVERSATIONAL FILLER PRUNING
# =========================================================
FILLER_PREFIXES = [
    r"(?i)^based on (the )?(provided )?(context|evidence|literature|documents)[,:]?\s*",
    r"(?i)^according to (the )?(provided )?(context|evidence|guidelines)[,:]?\s*",
    r"(?i)^the retrieved documents (state|indicate|suggest|show) that\s*",
    r"(?i)^in summary[,:]?\s*",
    r"(?i)^to answer your question[,:]?\s*",
    r"(?i)^as an oncology (assistant|expert|ai)[,:]?\s*",
]


def prune_conversational_filler(text: str) -> tuple[str, int]:
    """Removes verbose conversational preamble and filler intros."""
    if not text:
        return "", 0

    stripped = text
    count = 0
    for pattern in FILLER_PREFIXES:
        if re.search(pattern, stripped):
            stripped = re.sub(pattern, "", stripped).strip()
            count += 1

    return stripped, count


# =========================================================
# 🔹 MEDICAL SYNTAX & ENTITY POLISHING
# =========================================================
ONCOLOGY_ACRONYMS = {
    r"\begfr\b": "EGFR",
    r"\bnsclc\b": "NSCLC",
    r"\bsclc\b": "SCLC",
    r"\bkras\b": "KRAS",
    r"\bbraf\b": "BRAF",
    r"\balk\b": "ALK",
    r"\bher2\b": "HER2",
    r"\bher2-neu\b": "HER2/neu",
    r"\bpd-1\b": "PD-1",
    r"\bpdl1\b": "PD-L1",
    r"\bpd-l1\b": "PD-L1",
    r"\bbrca1\b": "BRCA1",
    r"\bbrca2\b": "BRCA2",
    r"\btnm\b": "TNM",
    r"\bstage 1\b": "Stage I",
    r"\bstage 2\b": "Stage II",
    r"\bstage 3\b": "Stage III",
    r"\bstage 4\b": "Stage IV",
    r"\bstage i\b": "Stage I",
    r"\bstage ii\b": "Stage II",
    r"\bstage iii\b": "Stage III",
    r"\bstage iv\b": "Stage IV",
}


def polish_medical_entities(text: str) -> tuple[str, int]:
    """Standardizes oncology abbreviations, capitalization, and clinical staging syntax."""
    if not text:
        return "", 0

    polished = text
    modifications = 0

    for pattern, replacement in ONCOLOGY_ACRONYMS.items():
        matches = len(re.findall(pattern, polished, re.IGNORECASE))
        if matches > 0:
            polished = re.sub(pattern, replacement, polished, flags=re.IGNORECASE)
            modifications += matches

    # Fix numbers missing percentage in survival rate contexts (e.g., 'rate of 40' -> 'rate of 40%')
    polished, p_sub = re.subn(r"(?i)(\bsurvival\s+rate\s+of\s+)(\d{1,2})\b(?!\s*%)", r"\1\2%", polished)
    modifications += p_sub

    return polished, modifications


# =========================================================
# 🔹 STRUCTURE & FORMATTING OPTIMIZATION
# =========================================================
def format_clinical_structure(text: str, query_type: str = "general", intent: str = "factual") -> str:
    """Standardizes bullet points, capitalizes leading terms, and polishes formatting."""
    if not text:
        return ""

    lines = text.split("\n")
    formatted_lines = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Standardize bullet markers to clean bullet point
        if re.match(r"^[-*•]\s*", line):
            line_content = re.sub(r"^[-*•]\s*", "", line).strip()
            if line_content:
                # Capitalize first letter
                line_content = line_content[0].upper() + line_content[1:] if len(line_content) > 1 else line_content.upper()
                formatted_lines.append(f"• {line_content}")
        elif re.match(r"^\d+[\.\)]\s*", line):
            # Numbered list
            formatted_lines.append(line)
        else:
            # Regular sentence paragraph
            if len(line) > 1:
                line = line[0].upper() + line[1:]
            formatted_lines.append(line)

    return "\n".join(formatted_lines).strip()


# =========================================================
# 🔹 VALIDATION & SAFETY GUARDRAILS
# =========================================================
def validate_optimized_answer(answer: str) -> bool:
    """Verifies that the optimized answer contains valid content without internal leakage."""
    if not answer or len(answer.strip()) < 2:
        return False

    answer_lower = answer.lower()
    for p in SEVERE_LEAK_PATTERNS:
        if p in answer_lower:
            return False

    return True


# =========================================================
# 🔹 MAIN OPTIMIZATION PIPELINE
# =========================================================
def optimize_response(
    raw_answer: str,
    query: str = "",
    docs: list = None,
    intent: str = "factual",
    query_type: str = "general",
    is_rag: bool = True
) -> dict:
    """
    Main entrypoint for the Response Optimization Layer.
    Transforms raw model output into a clinically structured, clean response with full diff analytics.
    """
    if not raw_answer or not str(raw_answer).strip():
        return {
            "optimized_answer": "Unable to generate a medical answer.",
            "raw_answer": raw_answer or "",
            "is_valid": False,
            "diagnostics": {
                "raw_chars": 0,
                "optimized_chars": 0,
                "reduction_percent": 0.0,
                "artifacts_stripped": 0,
                "duplicate_lines_removed": 0,
                "structure_type": query_type
            }
        }

    raw_text = str(raw_answer).strip()
    raw_len = len(raw_text)

    # 1. Artifact & Noise Stripping
    stripped_text, artifacts_count = strip_generation_artifacts(raw_text)

    # 2. Conversational Filler Pruning
    filler_pruned, filler_count = prune_conversational_filler(stripped_text)

    # 3. Deduplication & Redundancy Pruning
    pruned_text, dup_count = prune_redundancy(filler_pruned)

    # 4. Medical Syntax & Entity Polishing
    polished_text, entity_mod_count = polish_medical_entities(pruned_text)

    # 5. Structure & Clinical Formatting
    formatted_text = format_clinical_structure(polished_text, query_type=query_type, intent=intent)

    # 6. Validation
    is_valid = validate_optimized_answer(formatted_text)
    if not is_valid:
        final_answer = "Unable to generate a reliable medical answer."
    else:
        final_answer = formatted_text if formatted_text else "Unable to generate a medical answer."

    opt_len = len(final_answer)
    reduction_pct = round(((raw_len - opt_len) / max(raw_len, 1)) * 100, 1) if raw_len >= opt_len else 0.0

    diagnostics = {
        "raw_chars": raw_len,
        "optimized_chars": opt_len,
        "reduction_percent": reduction_pct,
        "artifacts_stripped": artifacts_count + filler_count,
        "duplicate_lines_removed": dup_count,
        "entities_polished": entity_mod_count,
        "structure_type": query_type,
        "is_rag": is_rag
    }

    return {
        "optimized_answer": final_answer,
        "raw_answer": raw_text,
        "is_valid": is_valid,
        "diagnostics": diagnostics
    }
