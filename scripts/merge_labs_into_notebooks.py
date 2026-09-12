#!/usr/bin/env python3
"""Merge the old Markdown labs into their corresponding notebooks.

Each lesson is now one learning unit: the blog explains the idea and the
matching notebook is the complete interactive lab. This script migrates the
old labs into notebooks, removes obsolete lab sections/links from blogs, and
adds one clear notebook/Colab navigation block to every blog.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

REPO = "manish7725/deeplearning"
BRANCH = "main"
BLOG_DIR = Path("blogs")
LAB_DIR = Path("labs")
NOTEBOOK_DIR = Path("notebooks")
MARKER = "<!-- LAB-MERGED-INTO-NOTEBOOK -->"
BLOG_MARKER = "<!-- NOTEBOOK-LAB-NAV -->"


def notebook_for_prefix(prefix: str) -> Path | None:
    matches = sorted(NOTEBOOK_DIR.glob(f"{prefix}-*.ipynb"))
    return matches[0] if matches else None


def lab_to_markdown(lab_text: str) -> str:
    lines = lab_text.splitlines()
    if lines and re.match(r"^#\s+", lines[0]):
        lines = lines[1:]
    body = "\n".join(lines).strip()
    return (
        f"{MARKER}\n\n"
        "## Hands-on Lab\n\n"
        "This is the original hands-on lab migrated from the old `labs/` area. "
        "The notebook is now the single place to run the experiment.\n\n"
        f"{body}\n"
    )


def merge_lab(lab_path: Path) -> tuple[str, Path] | None:
    match = re.match(r"^(\d{2})-", lab_path.name)
    if not match:
        return None

    prefix = match.group(1)
    notebook_path = notebook_for_prefix(prefix)
    if notebook_path is None:
        print(f"Skipping {lab_path}: no matching notebook")
        return None

    lab_text = lab_path.read_text(encoding="utf-8")
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    cells = notebook.setdefault("cells", [])

    existing = any(
        MARKER in "".join(cell.get("source", []))
        for cell in cells
        if cell.get("cell_type") == "markdown"
    )

    if not existing:
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in lab_to_markdown(lab_text).splitlines()],
        })
        notebook_path.write_text(
            json.dumps(notebook, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"Merged {lab_path} -> {notebook_path}")
    else:
        print(f"Already merged: {lab_path} -> {notebook_path}")

    lab_path.unlink()
    return prefix, notebook_path


def clean_old_lab_sections(text: str) -> str:
    # Remove the old standalone lab section because its content now lives in
    # the notebook. Keep later top-level sections such as external resources.
    text = re.sub(
        r"(?ms)^# 🧪 Hands-on Lab.*?(?=^# |\Z)",
        "",
        text,
    )
    # Remove any remaining direct links to the deleted labs directory.
    text = re.sub(r"^.*\]\([^\n]*?/labs/[^\n]*\)\s*\n", "", text, flags=re.MULTILINE)
    return text


def update_blog(prefix: str, notebook_path: Path) -> None:
    candidates = sorted(BLOG_DIR.glob(f"{prefix}-*.md"))
    if not candidates:
        print(f"No blog found for {prefix}")
        return

    blog_path = candidates[0]
    text = blog_path.read_text(encoding="utf-8")
    text = clean_old_lab_sections(text)

    # Replace the old per-code-block Colab banners with one navigation block.
    text = re.sub(
        r"^> 🧪 \*\*\[Run this code in Google Colab\]\([^\n]+\)\*\*\n\n",
        "",
        text,
        flags=re.MULTILINE,
    )

    notebook_url = f"https://github.com/{REPO}/blob/{BRANCH}/{notebook_path.as_posix()}"
    colab_url = (
        f"https://colab.research.google.com/github/{REPO}/blob/{BRANCH}/"
        f"{notebook_path.as_posix()}"
    )
    nav = (
        f"{BLOG_MARKER}\n\n"
        "## 🧪 Interactive Lab\n\n"
        "The explanation and the hands-on experiment now live together: "
        "the matching notebook is the complete runnable lab for this lesson.\n\n"
        f"**[📓 Open the notebook on GitHub]({notebook_url})**  "
        f"· **[▶ Open the notebook in Google Colab]({colab_url})**\n\n"
        "Run the cells, change the values, observe the result, and return to "
        "the blog to connect the experiment back to the idea.\n\n"
    )

    if BLOG_MARKER in text:
        start = text.index(BLOG_MARKER)
        rest = text[start:]
        next_heading = re.search(r"(?m)^## (?!🧪 Interactive Lab)", rest)
        suffix = rest[next_heading.start():] if next_heading else ""
        text = text[:start] + nav + suffix
    else:
        lines = text.splitlines(True)
        insert_at = 1 if lines and lines[0].lstrip().startswith("#") else 0
        text = "".join(lines[:insert_at]) + "\n" + nav + "".join(lines[insert_at:])

    blog_path.write_text(text, encoding="utf-8")
    print(f"Updated blog: {blog_path}")


def main() -> None:
    labs = sorted(LAB_DIR.glob("*.md"))
    if not labs:
        print("No labs remain; nothing to migrate.")
        return

    merged = []
    for lab_path in labs:
        result = merge_lab(lab_path)
        if result:
            merged.append(result)

    for prefix, notebook_path in merged:
        update_blog(prefix, notebook_path)

    print(f"Migrated {len(merged)} lab(s) into notebooks and removed obsolete lab files.")


if __name__ == "__main__":
    main()
