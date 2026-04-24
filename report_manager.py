import re

def create_report(episode_id: str, root_claim: str) -> str:
    """Returns initial Markdown Evidence Report string."""
    truncated_claim = root_claim[:100] + ("..." if len(root_claim) > 100 else "")
    return f"""## FORGE-MA Evidence Report — Episode {episode_id}
### Claim
{truncated_claim}

### Evidence Discovered
| Tool | Finding | Trust Signal | Primitive Hint |
|------|---------|--------------|----------------|

### Current Chain Hypothesis
GIN prediction: [? (unknown)]

### Budget
Steps used: 0/10 | Primitives found: 0 | Confidence: 0.00

### Next Recommended Tool
None
"""

def update_report(report: str, tool_name: str, finding: str,
                  trust_signal: str, gin_hint: str, steps_used: int = 1, budget_total: int = 10,
                  primitives_found: str = "0", confidence: str = "0.00", recommended_tool: str = "None") -> str:
    """Append tool result row to evidence table. Append GIN hint section."""
    lines = report.split('\n')
    
    # Find table end
    table_idx = -1
    for i, line in enumerate(lines):
        if line.startswith("### Current Chain Hypothesis"):
            table_idx = i - 1
            break
            
    if table_idx != -1:
        # Insert row
        lines.insert(table_idx, f"| {tool_name} | {finding} | {trust_signal} | {gin_hint.split()[0]} |")
        
    report = "\n".join(lines)
    
    # Update parts via regex
    report = re.sub(r'GIN prediction: .*', f'GIN prediction: {gin_hint}', report)
    report = re.sub(r'Steps used: \d+/\d+ \| Primitives found: [^\s]+ \| Confidence: [\d.]+',
                    f'Steps used: {steps_used}/{budget_total} | Primitives found: {primitives_found} | Confidence: {confidence}', report)
    report = re.sub(r'### Next Recommended Tool\n.*', f'### Next Recommended Tool\n{recommended_tool}', report)
    
    return report

def compress_report(report: str, max_tokens: int = 600) -> str:
    """Drop lowest-confidence rows when token count > 600. Preserve header."""
    if count_tokens(report) <= max_tokens:
        return report
    
    lines = report.split('\n')
    table_start = -1
    table_end = -1
    for i, line in enumerate(lines):
        if line.startswith("|------|"):
            table_start = i + 1
        elif line.startswith("### Current Chain Hypothesis"):
            table_end = i - 1
            break
            
    if table_start != -1 and table_end != -1 and table_end > table_start:
        rows = lines[table_start:table_end]
        if len(rows) > 0:
            # Drop the first (oldest or lowest confidence) row
            rows.pop(0)
            lines = lines[:table_start] + rows + lines[table_end:]
            return compress_report("\n".join(lines), max_tokens)
            
    return "\n".join(lines)

def extract_state(report: str) -> dict:
    """Parse report into structured dict for reward computation."""
    state = {}
    budget_match = re.search(r'Steps used: (\d+)/(\d+)', report)
    if budget_match:
        state['steps_used'] = int(budget_match.group(1))
        state['budget_total'] = int(budget_match.group(2))
    return state

def count_tokens(text: str) -> int:
    """Approximate: len(text.split()) * 1.3"""
    return int(len(text.split()) * 1.3)
