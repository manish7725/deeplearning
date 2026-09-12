import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BLOGS = ROOT / "blogs"
NOTEBOOKS = ROOT / "notebooks"
CHAPTERS = ROOT / "chapters"

# The 37 first-principles lessons are deliberately gated into five stages.
STAGES = {
    1: ("foundations-of-learning", "Foundations of Learning", range(1, 8)),
    2: ("mathematics-of-learning", "Mathematics of Learning", range(8, 13)),
    3: ("core-deep-learning", "Core Deep Learning", range(13, 21)),
    4: ("deep-learning-theory-and-modern-learning", "Deep Learning Theory & Modern Learning", range(21, 31)),
    5: ("advanced-deep-learning-and-research", "Advanced Deep Learning & Research", range(31, 38)),
}

STAGE_BY_LESSON = {
    lesson: stage
    for stage, (_, _, lesson_range) in STAGES.items()
    for lesson in lesson_range
}

LESSONS = sorted(
    p for p in BLOGS.glob("[0-9][0-9]-*.md") if 1 <= int(p.name[:2]) <= 37
)
if len(LESSONS) != 37:
    raise SystemExit(f"Expected 37 source blogs, found {len(LESSONS)}")


def title_for(blog):
    return next(
        (
            line[2:].strip()
            for line in blog.read_text(encoding="utf-8").splitlines()
            if line.startswith("# ")
        ),
        blog.stem,
    )


# Preserve the existing chapter folders while adding the new learner-facing stage hierarchy.
for blog in LESSONS:
    stem = blog.stem
    number = int(stem[:2])
    title = title_for(blog)
    stage = STAGE_BY_LESSON[number]
    stage_slug, stage_title, _ = STAGES[stage]

    legacy_folder = CHAPTERS / stem
    staged_root = CHAPTERS / f"stage-{stage}-{stage_slug}"
    folder = staged_root / stem
    apps = folder / "apps"
    folder.mkdir(parents=True, exist_ok=True)
    apps.mkdir(exist_ok=True)

    notebook = NOTEBOOKS / f"{stem}.ipynb"
    if not notebook.exists():
        raise SystemExit(f"Missing notebook for {stem}: {notebook}")

    blog_text = blog.read_text(encoding="utf-8")
    notebook_text = notebook.read_text(encoding="utf-8")

    (folder / f"{stem}.md").write_text(blog_text, encoding="utf-8")
    (folder / f"{stem}.ipynb").write_text(notebook_text, encoding="utf-8")

    readme = f"""# Chapter {number:02d} — {title}

**Stage {stage}: {stage_title}**

## Start here

📓 **[Open the interactive lesson](./{stem}.ipynb)**

The notebook is the primary learning experience: story → intuition → mathematics → hand calculation → NumPy → PyTorch → visualization → controlled experiment → failure mode → exercises → application → research bridge.

## Source article

📖 **[Read the parallel Markdown source](./{stem}.md)**

The Markdown copy is retained for authoring, publishing, search, provenance, and synchronization. The notebook is intentionally self-contained.

## Applications

🛠️ **[Chapter applications](./apps/)**

Applications belong beside the lesson and should demonstrate the chapter's mathematical idea rather than hide it behind a framework.

## Progression

> Complete the earlier chapters in **Stage {stage}** before treating this chapter as mastered.

The stage must be passed before moving to Stage {stage + 1 if stage < 5 else stage}.

## Source mapping

- Canonical source: `blogs/{stem}.md`
- Learner notebook: `chapters/stage-{stage}-{stage_slug}/{stem}/{stem}.ipynb`
- Parallel source copy: `chapters/stage-{stage}-{stage_slug}/{stem}/{stem}.md`
- Applications: `chapters/stage-{stage}-{stage_slug}/{stem}/apps/`
"""
    (folder / "README.md").write_text(readme, encoding="utf-8")

    app_readme = f"""# Applications — Chapter {number:02d}

Build small applications using the ideas from **{title}**.

Keep application code separate from the lesson notebook. Each application should include runnable code and a short explanation of the mathematical concept it demonstrates.
"""
    (apps / "README.md").write_text(app_readme, encoding="utf-8")


# Stage-level landing pages.
for stage, (slug, stage_title, lesson_range) in STAGES.items():
    stage_root = CHAPTERS / f"stage-{stage}-{slug}"
    rows = [
        f"# Stage {stage} — {stage_title}",
        "",
        "This is a **gated stage**. Complete the chapters in order and pass the stage mastery gate before progressing.",
        "",
        "| # | Chapter | Interactive lesson | Applications |",
        "|---:|---|---|---|",
    ]
    for number in lesson_range:
        blog = next(p for p in LESSONS if int(p.stem[:2]) == number)
        stem = blog.stem
        title = title_for(blog)
        rows.append(
            f"| {number:02d} | {title} | [Open notebook](./{stem}/{stem}.ipynb) | [Apps](./{stem}/apps/) |"
        )
    rows += [
        "",
        "## Stage gate",
        "",
        "See [`STAGES.md`](../../STAGES.md) for the stage objective and mastery gate.",
    ]
    (stage_root / "README.md").write_text("\n".join(rows) + "\n", encoding="utf-8")


index = [
    "# Five-Stage Deep Learning Journey",
    "",
    "The repository is a **gated learning path**, not a flat list of tutorials. The notebook is the primary interactive textbook and laboratory; Markdown remains the parallel source.",
    "",
    "> **Finish Stage 1 → pass its gate → Stage 2 → pass its gate → ... → Stage 5.**",
    "",
    "| Stage | Focus | Chapters |",
    "|---:|---|---:|",
]
for stage, (_, stage_title, lesson_range) in STAGES.items():
    first, last = min(lesson_range), max(lesson_range)
    slug = STAGES[stage][0]
    index.append(
        f"| {stage} | [{stage_title}](./stage-{stage}-{slug}/) | {first:02d}–{last:02d} |"
    )
index += [
    "",
    "## Stage folders",
    "",
    "Each stage contains its chapter folders. Every chapter contains the notebook, Markdown source, README, and applications directory.",
    "",
    "```text",
    "chapters/",
    "├── stage-1-foundations-of-learning/",
    "│   ├── 01-what-is-learning/",
    "│   │   ├── 01-what-is-learning.ipynb  ← START HERE",
    "│   │   ├── 01-what-is-learning.md",
    "│   │   ├── README.md",
    "│   │   └── apps/",
    "│   └── ...",
    "├── stage-2-mathematics-of-learning/",
    "├── stage-3-core-deep-learning/",
    "├── stage-4-deep-learning-theory-and-modern-learning/",
    "└── stage-5-advanced-deep-learning-and-research/",
    "```",
    "",
    "See [`STAGES.md`](../STAGES.md) for prerequisites, goals, and mastery gates.",
]
(CHAPTERS / "README.md").write_text("\n".join(index) + "\n", encoding="utf-8")
print("Organized 37 chapters into five gated stages.")
