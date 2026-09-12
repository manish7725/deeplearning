import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BLOGS = ROOT / "blogs"
NOTEBOOKS = ROOT / "notebooks"
CHAPTERS = ROOT / "chapters"

LESSONS = sorted(
    p for p in BLOGS.glob("[0-9][0-9]-*.md") if 1 <= int(p.name[:2]) <= 37
)
if len(LESSONS) != 37:
    raise SystemExit(f"Expected 37 source blogs, found {len(LESSONS)}")

for blog in LESSONS:
    stem = blog.stem
    number = stem[:2]
    folder = CHAPTERS / stem
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "apps").mkdir(exist_ok=True)

    notebook = NOTEBOOKS / f"{stem}.ipynb"
    if not notebook.exists():
        raise SystemExit(f"Missing notebook for {stem}: {notebook}")

    # Keep the original source files intact. Chapter folders are the learner-facing workspace.
    (folder / f"{stem}.md").write_text(blog.read_text(encoding="utf-8"), encoding="utf-8")
    (folder / f"{stem}.ipynb").write_text(notebook.read_text(encoding="utf-8"), encoding="utf-8")

    data = json.loads(notebook.read_text(encoding="utf-8"))
    title = next(
        (
            line[2:].strip()
            for line in blog.read_text(encoding="utf-8").splitlines()
            if line.startswith("# ")
        ),
        stem,
    )

    readme = f"""# Chapter {number} — {title}\n\nThis is the learner-facing workspace for this chapter.\n\n## Start here\n\n📓 **[Open the interactive lesson](./{stem}.ipynb)**\n\nThe notebook is the primary learning experience: explanation → mathematics → hand calculation → NumPy → PyTorch → visualization → experiment → failure → exercises → project → research bridge.\n\n## Source article\n\n📖 **[Read the parallel Markdown source](./{stem}.md)**\n\nThe Markdown source is retained for editing, publishing, search, and provenance. The notebook is intentionally self-contained so a Class 8 student can learn without opening the Markdown file.\n\n## Applications\n\n🛠️ **[Chapter applications](./apps/)**\n\nUse this directory for applications and small projects that grow naturally from this chapter. Keep application code separate from the core lesson notebook.\n\n## Source mapping\n\n- Canonical source: `blogs/{stem}.md`\n- Learner notebook: `chapters/{stem}/{stem}.ipynb`\n- Parallel source copy: `chapters/{stem}/{stem}.md`\n- Applications: `chapters/{stem}/apps/`\n\n## Learning rule\n\n> **Start with the notebook.** The notebook is the interactive textbook and laboratory for this chapter.\n"""
    (folder / "README.md").write_text(readme, encoding="utf-8")

    app_readme = f"""# Applications — Chapter {number}\n\nBuild small applications here that use the ideas from **{title}**.\n\nKeep the application focused on the chapter's core concept. Each application should include its own README, runnable code, and a short explanation of what mathematical idea it demonstrates.\n"""
    (folder / "apps" / "README.md").write_text(app_readme, encoding="utf-8")

index_lines = [
    "# Interactive Deep Learning Chapters",
    "",
    "The **notebook is the primary learning experience**. Each chapter is self-contained: theory, mathematics, code, visualization, experiments, exercises, projects, and research connection.",
    "",
    "```text",
    "Chapter workspace",
    "├── README.md              ← chapter map",
    "├── NN-topic.ipynb         ← START HERE: interactive textbook + laboratory",
    "├── NN-topic.md            ← parallel Markdown source",
    "└── apps/                  ← applications built from the chapter",
    "```",
    "",
    "| # | Chapter | Interactive lesson | Applications |",
    "|---:|---|---|---|",
]
for blog in LESSONS:
    stem = blog.stem
    title = next((line[2:].strip() for line in blog.read_text(encoding="utf-8").splitlines() if line.startswith("# ")), stem)
    index_lines.append(
        f"| {stem[:2]} | {title} | [Open notebook](./{stem}/{stem}.ipynb) | [Apps](./{stem}/apps/) |"
    )
(CHAPTERS / "README.md").write_text("\n".join(index_lines) + "\n", encoding="utf-8")
print(f"Organized {len(LESSONS)} chapters under chapters/")
