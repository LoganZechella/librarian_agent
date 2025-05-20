"""
Response formatting and markdown rendering for Librarian Agent.
"""
from .schema import AgentOutput, ResultItem # Import AgentOutput and ResultItem
from typing import List # Import List for type hinting if needed for internal vars

def render_markdown(resp: AgentOutput) -> str: # Changed resp type to AgentOutput
    md = f"## Summary\n{resp.summary}\n\n" # Direct attribute access
    md += "## Results\n"
    # resp.results is already List[ResultItem]
    for r in resp.results:
        # Direct attribute access for ResultItem fields
        citation_str = str(r.citation_id) # Ensure string for f-string
        page_str = str(r.page) if r.page is not None else "?"
        md += f"- [{citation_str}] {r.excerpt} _(p.{page_str} | {r.source})_\n"
    md += "\n## Next Steps\n"
    # resp.next_steps is already List[str]
    md += "\n".join(f"- {s}" for s in resp.next_steps)
    return md
