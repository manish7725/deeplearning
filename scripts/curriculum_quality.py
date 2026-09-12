from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BLOGS = ROOT / "blogs"
NOTEBOOKS = ROOT / "notebooks"


def number(path: Path) -> int:
    match = re.match(r"(\d+)-", path.name)
    return int(match.group(1)) if match else 10_000


def notebook_title(path: Path) -> str:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        for cell in data.get("cells", []):
            if cell.get("cell_type") == "markdown":
                text = "".join(cell.get("source", []))
                match = re.search(r"^#\s+(.+)$", text, re.M)
                if match:
                    return match.group(1).strip()
    except Exception:
        return ""
    return ""


def main() -> int:
    blogs = sorted(BLOGS.glob("*.md"), key=number)
    notebooks = sorted(NOTEBOOKS.glob("*.ipynb"), key=number)
    nb_by_number = {number(p): p for p in notebooks}

    failures: list[str] = []
    warnings: list[str] = []

    if not blogs:
        failures.append("No blog files found")
    if not notebooks:
        failures.append("No notebook files found")

    for blog in blogs:
        n = number(blog)
        text = blog.read_text(encoding="utf-8")
        if n not in nb_by_number:
            failures.append(f"{blog.name}: missing matching notebook")
        for required, pattern in {
            "learning bridge": r"(?i)(previous|came from|next|where.*go|continue)",
            "math": r"(?i)(equation|derivative|gradient|matrix|vector|loss|probability|function|formula)",
        }.items():
            if not re.search(pattern, text):
                warnings.append(f"{blog.name}: weak/missing {required} signal")

    for nb in notebooks:
        try:
            data = json.loads(nb.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            failures.append(f"{nb.name}: invalid JSON ({exc})")
            continue
        cells = data.get("cells", [])
        if not cells:
            failures.append(f"{nb.name}: notebook has no cells")
        if not any(c.get("cell_type") == "code" for c in cells):
            warnings.append(f"{nb.name}: no executable code cell")
        if not notebook_title(nb):
            warnings.append(f"{nb.name}: no markdown H1 title")

    report = [
        "# Curriculum Quality Report",
        "",
        f"Blogs: {len(blogs)}",
        f"Notebooks: {len(notebooks)}",
        f"Failures: {len(failures)}",
        f"Warnings: {len(warnings)}",
        "",
        "## Failures",
        *([f"- {x}" for x in failures] or ["- None"]),
        "",
        "## Warnings",
        *([f"- {x}" for x in warnings] or ["- None"]),
    ]
    (ROOT / "CURRICULUM_QUALITY.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
