#!/usr/bin/env python3
"""Merge the old Markdown labs into their corresponding notebooks.

The repository now treats each notebook as the single hands-on lab for its
blog post. This script:
  1. Appends each labs/*.md lab as a Markdown section in the matching notebook.
  2. Adds a small navigation block to the matching blog post with links to
     the notebook source and Google Colab.
  3. Removes the obsolete labs/*.md files.

It is intentionally idempotent so the GitHub Actions workflow can run safely
on every push.
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
    # Keep the lab's useful teaching material, but avoid another top-level
    # notebook title inside the notebook.
    lines = lab_text.splitlines()
    if lines and re.match(r"^#\s+", lines[0]):
        lines = lines[1:]
    body = "\n".join(lines).strip()
    return (
        f"{MARKER}\n\n"
        "## Hands-on Lab\n\n"
        "This section contains the original hands-on lab for this lesson. "
        "The notebook is now the single place to run the experiment.\n\n"
        f"{body}\n"
    )


def merge_lab(lab_path: Path) -> Path | None:
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
        cells.append(
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [line + "\n" for line in lab_to_markdown(lab_text).splitlines()],
            }
        )

        notebook_path.write_text(
            json.dumps(notebook, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"Merged {lab_path} -> {notebook_path}")
    else:
        print(f"Already merged: {lab_path} -> {notebook_path}")

    lab_path.unlink()
    return notebook_path


def update_blog(prefix: str, notebook_path: Path) -> None:
    candidates = sorted(BLOG_DIR.glob(f"{prefix}-*.md"))
    if not candidates:
        print(f"No blog found for {prefix}")
        return

    blog_path = candidates[0]
    text = blog_path.read_text(encoding="utf-8")

    # Remove the older per-code-block Colab banners. One clear notebook link
    # at the beginning is easier to follow and avoids navigation noise.
    text = re.sub(
        r"^> 🧪 \*\*\[Run this code in Google Colab\]\([^\n]+\)\*\*\n\n",
        "",
        text,
        flags=re.MULTILINE,
    )

    notebook_url = (
        f"https://github.com/{REPO}/blob/{BRANCH}/{notebook_path.as_posix()}"
    )
    colab_url = (
        f"https://colab.research.google.com/github/{REPO}/blob/{BRANCH}/"
        f"{notebook_path.as_posix()}"
    )

    nav = (
        f"{BLOG_MARKER}\n\n"
        "## 🧪 Interactive Lab\n\n"
        "The explanation and the hands-on experiment now live together: "
        "the notebook contains the complete runnable lab for this lesson.\n\n"
        f"**[📓 Open the notebook on GitHub]({notebook_url})**  "
        f"· **[▶ Open the notebook in Google Colab]({colab_url})**\n\n"
        "Run the cells, change the values, observe the result, and then return "
        "to this blog to connect the experiment back to the idea.\n\n"
    )

    # Replace an existing navigation block if the workflow is re-run.
    if BLOG_MARKER in text:
        text = re.sub(
            rf"{re.escape(BLOG_MARKER)}.*?(?=^# |^## |\Z)",
            nav,
            text,
            flags=re.MULTILINE | re.DOTALL,
        )
    else:
        lines = text.splitlines(True)
        insert_at = 1 if lines and lines[0].lstrip().startswith("#") else 0
        text = "".join(lines[:insert_at]) + "\n" + nav + "".join(lines[insert_at:])

    blog_path.write_text(text, encoding="utf-8")
    print(f"Updated blog navigation: {blog_path}")


def main() -> None:
    LAB_DIR.mkdir(exist_ok=True)
    NOTEBOOK_DIR.mkdir(exist_ok=True)
    BLOG_DIR.mkdir(exist_ok=True)

    labs = sorted(LAB_DIR.glob("*.md"))
    if not labs:
        print("No labs remain; nothing to merge.")
        return

    merged = []
    for lab_path in labs:
        notebook_path = merge_lab(lab_path)
        if notebook_path:
            prefix = notebook_path.name[:2]
            merged.append((prefix, notebook_path))

    for prefix, notebook_path in merged:
        update_blog(prefix, notebook_path)

    print(f"Merged {len(merged)} lab(s) into notebooks and removed obsolete lab files.")


if __name__ == "__main__":
    main()
