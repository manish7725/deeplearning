from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
BLOGS = ROOT / "blogs"
NOTEBOOKS = ROOT / "notebooks"


def numbered(path):
    m = re.match(r"(\d+)-", path.name)
    return int(m.group(1)) if m else 10_000


def title_from_markdown(text, fallback):
    for line in text.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return fallback


def main():
    blogs = sorted(BLOGS.glob("*.md"), key=numbered)
    notebooks = sorted(NOTEBOOKS.glob("*.ipynb"), key=numbered)

    notebook_by_num = {numbered(p): p for p in notebooks}
    rows = []
    missing_notebooks = []
    weak_lessons = []

    for blog in blogs:
        n = numbered(blog)
        text = blog.read_text(encoding="utf-8")
        title = title_from_markdown(text, blog.stem)
        nb = notebook_by_num.get(n)
        if not nb:
            missing_notebooks.append(f"{n:02d} {blog.name}")
        checks = {
            "objective": bool(re.search(r"objective|goal|learn", text, re.I)),
            "math": bool(re.search(r"derivative|equation|matrix|vector|loss|gradient|probability", text, re.I)),
            "next": bool(re.search(r"next|where.*go|continue", text, re.I)),
        }
        if sum(checks.values()) < 2:
            weak_lessons.append(f"{n:02d} {blog.name}")
        nb_name = nb.name if nb else "—"
        colab = f"https://colab.research.google.com/github/manish7725/deeplearning/blob/main/notebooks/{nb.name}" if nb else "—"
        rows.append(f"| {n:02d} | [{title}](blogs/{blog.name}) | [{nb_name}](notebooks/{nb_name}) | [Open in Colab]({colab}) |")

    report = "# Learning Map\n\nGenerated automatically. The blog is the textbook; the notebook is the laboratory.\n\n## Lesson map\n\n| # | Blog | Notebook | Colab |\n|---:|---|---|---|\n" + "\n".join(rows) + "\n\n## Automatic audit\n\n"
    report += f"- Blogs discovered: **{len(blogs)}**\n"
    report += f"- Notebooks discovered: **{len(notebooks)}**\n"
    report += f"- Blogs without a matching notebook: **{len(missing_notebooks)}**\n"
    report += f"- Blogs needing content review: **{len(weak_lessons)}**\n"
    if missing_notebooks:
        report += "\n### Missing notebooks\n\n" + "\n".join(f"- {x}" for x in missing_notebooks) + "\n"
    if weak_lessons:
        report += "\n### Content review queue\n\n" + "\n".join(f"- {x}" for x in weak_lessons) + "\n"
    report += "\n## Learning contract\n\nStory → intuition → numerical example → mathematics → hand calculation → NumPy → PyTorch → visualization → experiment → failure mode → project → paper → research question.\n"

    (ROOT / "LEARNING_MAP.md").write_text(report, encoding="utf-8")

    # A machine-readable audit artifact for local use and CI logs.
    audit = ROOT / "learning_audit.txt"
    audit.write_text(
        "\n".join([
            f"blogs={len(blogs)}",
            f"notebooks={len(notebooks)}",
            f"missing_notebooks={len(missing_notebooks)}",
            f"content_review_queue={len(weak_lessons)}",
        ]) + "\n", encoding="utf-8"
    )
    print(report)


if __name__ == "__main__":
    main()
