"""
Response formatting and markdown rendering for Librarian Agent.
"""

def render_markdown(resp: dict) -> str:
    md = f"## Summary\n{resp.get('summary', '')}\n\n"
    md += "## Results\n"
    for r in resp.get("results", []):
        md += f"- [{r.get('citation_id', '?')}] {r.get('excerpt', '')} _(p.{r.get('page', '?')} | {r.get('source', '')})_\n"
    md += "\n## Next Steps\n"
    md += "\n".join(f"- {s}" for s in resp.get("next_steps", []))
    return md
