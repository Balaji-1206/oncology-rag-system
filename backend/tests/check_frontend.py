"""Checks frontend HTML for hardcoded metrics and dynamic data patterns."""

with open(r"c:\Users\Sandhiya P\NIT INTERN\oncology-agentic-rag\frontend\index.html", encoding="utf-8") as f:
    content = f.read()

lower = content.lower()

print("=== FRONTEND VERIFICATION REPORT ===\n")
print(f"File size: {len(content)} bytes\n")

checks = {
    "API fetch() call": "fetch(",
    "response.json()": "response.json",
    "disclaimer text": "research and educational",
    "clinical warning": "not certified",
    "grounding_score key": "grounding_score",
    "evaluation key": "evaluation",
    "sources key": "sources",
    "textContent (safe DOM write)": "textcontent",
    "innerHTML usage": "innerhtml",
    "innerHTML sanitized": "innerhtml",
    "512 static in JS": "512",
    "0.87 static": "0.87",
    "0.60 static": "0.60",
    "Hardcoded score": "score: 9",
    "fetchSettings call": "fetchsettings",
    "fetchDiagnostics call": "fetchdiagnostics",
    "Backend disclaimer from API": "disclaimer",
    "escapeHtml function": "escapehtml",
    "DOMPurify or escape fn": "escape",
    "ARIA labels": "aria-label",
    "keyboard nav": "tabindex",
    "mobile responsive": "mobile",
}

for label, keyword in checks.items():
    found = keyword in lower
    count = lower.count(keyword)
    status = "YES" if found else "NO"
    print(f"  {label:<40s}: {status}  (count={count})")

print()
print("--- Raw 512 occurrences in JS context ---")
lines = content.split("\n")
for i, line in enumerate(lines, 1):
    if "512" in line and "var" not in line.lower() and "#" not in line:
        print(f"  L{i}: {line.strip()[:120]}")
