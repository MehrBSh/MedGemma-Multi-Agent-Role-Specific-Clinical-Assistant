# ============================================================
# Export saved notes to Markdown (and optionally PDF).
# ============================================================

from pathlib import Path
from datetime import datetime

import store


def export_to_markdown(filepath: str | Path | None = None) -> str:
    """Export all notes to a Markdown file. Returns path written. Default: notes_export_YYYYMMDD.md."""
    if filepath is None:
        filepath = Path(__file__).parent / f"notes_export_{datetime.now().strftime('%Y%m%d_%H%M')}.md"
    filepath = Path(filepath)
    notes = store.get_all_notes()
    lines = [
        "# My learning notes",
        f"*Exported {datetime.now().isoformat()}*",
        "",
    ]
    for i, n in enumerate(notes, 1):
        lines.append(f"## Note {i}")
        lines.append(f"**Q:** {n['question']}")
        lines.append("")
        lines.append(f"**A:** {n['answer']}")
        if n.get("sources"):
            lines.append("")
            lines.append("**Sources:**")
            for s in n["sources"][:3]:
                lines.append(f"- {s[:200]}..." if len(s) > 200 else f"- {s}")
        lines.append("")
        lines.append("---")
        lines.append("")
    filepath.write_text("\n".join(lines), encoding="utf-8")
    return str(filepath)
