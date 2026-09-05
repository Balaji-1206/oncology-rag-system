"""
Response Optimization Layer — Ablation & Diff Showcase Tool
============================================================
This script demonstrates the measurable clinical value and impact of the
Response Optimization Layer by running an Ablation Study comparing:
1. Without Response Optimization (Raw unconstrained generation with reasoning leaks, filler, and formatting noise)
2. With Response Optimization (Clinically formatted, noise-stripped, and standardized output)
"""

import sys
import os
import argparse

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add backend directory to sys.path
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from modules.optimization.response_optimizer import optimize_response


BENCHMARK_ABLATION_CASES = [
    {
        "id": "CASE-1",
        "title": "Reasoning & <think> Tag Leakage (CoT Model)",
        "query": "What are the targeted therapy options for EGFR mutated NSCLC?",
        "query_type": "treatment",
        "intent": "clinical_guidance",
        "raw_answer": """<think>
The user is asking for targeted therapy options for EGFR mutated non-small cell lung cancer.
Let me recall from the provided context:
First generation TKIs include gefitinib and erlotinib.
Second generation includes afatinib.
Third generation includes osimertinib.
I will formulate the bullet points directly.
</think>
Reasoning: The primary targeted therapy options are tyrosine kinase inhibitors.
- gefitinib
- erlotinib
- afatinib
- osimertinib
- gefitinib is indicated for adenocarcinoma
Observation: All four drugs are FDA approved for EGFR mutations."""
    },
    {
        "id": "CASE-2",
        "title": "Conversational Filler & Verbose Preamble",
        "query": "What is the basic surgical treatment for testicular cancer?",
        "query_type": "treatment",
        "intent": "clinical_guidance",
        "raw_answer": """Based on the provided literature and clinical oncology guidelines, to answer your question regarding testicular neoplasms:
The basic surgical treatment is always radical inguinal orchiectomy, performed within 24-48 hours after diagnosis.
In summary, this represents the standard surgical management."""
    },
    {
        "id": "CASE-3",
        "title": "Medical Acronym & Staging Syntax Normalization",
        "query": "What does a 5-year survival rate of ~40% indicate in colorectal cancer?",
        "query_type": "prognosis",
        "intent": "factual",
        "raw_answer": """survival rate of 40 in colorectal cancer generally indicates disease with nodal positivity, corresponding to stage 3 or stage iii disease.
patients with stage 3 colorectal cancer have regional lymph node involvement."""
    },
    {
        "id": "CASE-4",
        "title": "Redundancy & Repetitive Chunk Collapse",
        "query": "What are common symptoms of right-sided colon tumors?",
        "query_type": "symptoms",
        "intent": "factual",
        "raw_answer": """- Indeterminate abdominal pain and pressure
- Fatigue and weakness due to anemia from chronic occult blood loss
- Indeterminate abdominal pain and discomfort in right quadrant
- Fatigue and generalized weakness
- Palpable right lower quadrant abdominal mass"""
    },
    {
        "id": "CASE-5",
        "title": "Clinical Definition & Prompt Echo Sanitization",
        "query": "What defines a Krukenberg tumor?",
        "query_type": "definition",
        "intent": "factual",
        "raw_answer": """The user wants a clear definition of Krukenberg tumor.
FINAL ANSWER:
A Krukenberg tumor is defined as a metastatic signet-ring cell carcinoma of the ovary, most commonly originating from a primary gastrointestinal malignancy (typically the stomach)."""
    }
]


def display_comparison_card(case: dict, index: int, total: int):
    """Renders a boxed side-by-side terminal comparison card."""
    raw_answer = case["raw_answer"]
    query = case["query"]
    query_type = case.get("query_type", "general")
    intent = case.get("intent", "factual")

    opt_result = optimize_response(
        raw_answer=raw_answer,
        query=query,
        intent=intent,
        query_type=query_type,
        is_rag=True
    )

    opt_answer = opt_result["optimized_answer"]
    diag = opt_result["diagnostics"]

    print("\n" + "═" * 72)
    print(f"  🧪 TEST CASE {index}/{total}: {case['title']}")
    print(f"  ❓ Medical Query: {query}")
    print(f"  📂 Category     : {query_type.capitalize()}  |  Intent: {intent.capitalize()}")
    print("═" * 72)

    # 1. Without Optimization
    print("\n  ❌ [1] Without Response Optimization (Raw LLM Output):")
    print("  " + "┄" * 68)
    for line in raw_answer.strip().splitlines():
        print(f"  │  {line}")

    # 2. With Optimization
    print("\n  ✨ [2] With Response Optimization (Refined Clinical Output):")
    print("  " + "┄" * 68)
    for line in opt_answer.strip().splitlines():
        print(f"  │  {line}")

    # 3. Quantitative Diagnostics & Metrics
    print("\n  📊 [3] Optimization Layer Diagnostics (Ablation Metrics):")
    print("  " + "┄" * 68)
    print(f"  • Size Delta          : {diag['raw_chars']} chars → {diag['optimized_chars']} chars ({diag['reduction_percent']}% reduction)")
    print(f"  • Artifacts Stripped  : {diag['artifacts_stripped']} noise patterns (<think>, CoT reasoning, prompt echoes)")
    print(f"  • Duplicates Removed  : {diag['duplicate_lines_removed']} redundant/repeating lines")
    print(f"  • Entities Polished   : {diag.get('entities_polished', 0)} oncology acronym/staging syntax normalizations")
    print(f"  • Validation Status   : {'PASSED' if opt_result['is_valid'] else 'FLAGGED'} (Clinical safety & prompt leak verification)")


def run_interactive_mode():
    """Allows user to input raw text or queries and see instant optimization."""
    print("\n" + "═" * 72)
    print("  🧠 Interactive Response Optimizer Diff Tool")
    print("  Type 'exit' to quit")
    print("═" * 72)

    while True:
        try:
            print("\nEnter sample RAW LLM text (or type a query to test):")
            raw_input = input("> ").strip()
            if raw_input.lower() in ["exit", "quit"]:
                break
            if not raw_input:
                continue

            opt_result = optimize_response(raw_input)
            diag = opt_result["diagnostics"]

            print("\n" + "─" * 66)
            print("  ✨ Cleaned & Optimized Clinical Answer:")
            print("─" * 66)
            print(opt_result["optimized_answer"])

            print("\n  📊 Optimization Stats:")
            print(f"  • Delta: {diag['raw_chars']} → {diag['optimized_chars']} chars ({diag['reduction_percent']}% reduction)")
            print(f"  • Artifacts Stripped: {diag['artifacts_stripped']}")
            print(f"  • Duplicates Removed: {diag['duplicate_lines_removed']}")
            print(f"  • Entities Polished: {diag.get('entities_polished', 0)}")
            print("─" * 66)

        except (KeyboardInterrupt, EOFError):
            break


def main():
    parser = argparse.ArgumentParser(description="Response Optimization Layer Ablation & Diff Showcase")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive text optimization tester")
    args = parser.parse_args()

    if args.interactive:
        run_interactive_mode()
        return

    print("\n" + "═" * 72)
    print("  🔬 Oncology Agentic RAG — Response Optimization Layer Ablation Study")
    print("  Demonstrating real-world diffs across 5 realistic clinical generation cases")
    print("═" * 72)

    for idx, case in enumerate(BENCHMARK_ABLATION_CASES, 1):
        display_comparison_card(case, idx, len(BENCHMARK_ABLATION_CASES))

    print("\n" + "═" * 72)
    print("  ✅ Ablation Demonstration Completed Successfully")
    print("  To run interactive custom testing: python demo_optimization_diff.py --interactive")
    print("═" * 72 + "\n")


if __name__ == "__main__":
    main()
