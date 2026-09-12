"""Class-8-student walkthrough fixes for blogs and notebooks.

The notebook is the laboratory for the blog, so every lesson gets an explicit
bridge from the previous lesson to the current one and then to the next one.
A few early lessons also get an exact executable version of the blog's anchor
example so a beginner is not forced to translate between unrelated examples.
"""
from pathlib import Path
import json
import re

ROOT = Path(__file__).resolve().parents[1]
BLOGS = ROOT / "blogs"
NOTEBOOKS = ROOT / "notebooks"


def blog_title(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    m = re.search(r"^# Blog \d{2} — (.+)$", text, re.M)
    return m.group(1).strip() if m else path.stem


def bridge(prefix: str, current: str, previous: str | None, following: str | None) -> str:
    prev = f"**Previous lesson:** Blog {int(prefix)-1:02d} — {previous}.\n\n" if previous else "This is the starting point of the course.\n\n"
    nxt = f"**Next lesson:** Blog {int(prefix)+1:02d} — {following}." if following else "You have reached the end of the current learning path."
    return (
        "## 🧭 Where this lesson fits\n\n"
        f"{prev}"
        f"**Today:** Blog {prefix} — {current}.\n\n"
        f"{nxt}\n\n"
        "**Student rule:** if you cannot explain why this lesson follows the previous one, stop and reread the final takeaway of the previous blog. "
        "The equations below should feel like a continuation, not a new language."
    )


def exact_anchor(prefix: str) -> tuple[str, str] | None:
    if prefix == "01":
        return (
            "## 🔗 Exact example from Blog 01\n\n"
            "The blog asks us to learn the simple rule `y = 2x + 1`. We will use exactly that example here. "
            "First we predict with `y_hat = w*x + b`; then we measure the error; finally we update `w` and `b`.",
            """x = np.array([1., 2., 3., 4.])
y = np.array([3., 5., 7., 9.])
w, b = 0.0, 0.0
learning_rate = 0.01

for step in range(1000):
    y_hat = w * x + b
    error = y_hat - y
    loss = np.mean(error ** 2)
    dw = np.mean(2 * error * x)
    db = np.mean(2 * error)
    w -= learning_rate * dw
    b -= learning_rate * db

print(f"learned w={w:.3f}, b={b:.3f}")
print(f"prediction for x=5: {w*5+b:.3f}")""",
        )
    if prefix == "03":
        return (
            "## 🔗 Exact example from Blog 03\n\n"
            "The blog says matrix multiplication is a **grid of dot products**. Here we calculate the same numbers by hand in Python so you can see where every output entry comes from.",
            """X = np.array([[2., 3.], [4., 5.]])
W = np.array([[10.], [20.]])

# First row dot product: 2*10 + 3*20 = 80
# Second row dot product: 4*10 + 5*20 = 140
print(X @ W)
print("first output by hand:", 2*10 + 3*20)
print("second output by hand:", 4*10 + 5*20)""",
        )
    if prefix == "08":
        return (
            "## 🔗 Exact example from Blog 08\n\n"
            "The blog connects calculus to learning with `L(w) = (w-3)^2`, whose derivative is `dL/dw = 2(w-3)`. "
            "We verify the same idea numerically and with PyTorch.",
            """def loss(w):
    return (w - 3.0) ** 2

w = 5.0
h = 1e-5
numerical = (loss(w + h) - loss(w)) / h
analytical = 2 * (w - 3.0)
print("numerical derivative:", numerical)
print("analytical derivative:", analytical)

import torch
wt = torch.tensor(5.0, requires_grad=True)
lt = (wt - 3.0) ** 2
lt.backward()
print("PyTorch derivative:", wt.grad.item())""",
        )
    return None


def patch_blog(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    # Fix the known Blog 03 malformed code fence/copy-paste section.
    if path.name.startswith("03-"):
        text = re.sub(
            r"\n# Manually verify one element\.\nprint\(2\*10 \+ 3\*20\)\n```\n\n### Challenges",
            "\n\n### Challenges",
            text,
        )
    # Add a beginner-facing bridge once, immediately after the notebook nav block.
    if "## 🧭 Where this lesson fits" not in text:
        prefix = path.name[:2]
        paths = sorted(BLOGS.glob(f"{prefix}-*.md"))
        current = blog_title(path)
        all_blogs = sorted(BLOGS.glob("*.md"))
        idx = all_blogs.index(path)
        previous = blog_title(all_blogs[idx - 1]) if idx else None
        following = blog_title(all_blogs[idx + 1]) if idx + 1 < len(all_blogs) else None
        block = bridge(prefix, current, previous, following)
        marker = "**[▶ Open the notebook in Google Colab]"
        pos = text.find("\n\n", text.find(marker))
        if pos != -1:
            text = text[:pos] + "\n\n" + block + text[pos:]
    path.write_text(text, encoding="utf-8")
    return True


def patch_notebook(path: Path, blog_paths: list[Path]) -> bool:
    nb = json.loads(path.read_text(encoding="utf-8"))
    prefix = path.name[:2]
    matching = next((p for p in blog_paths if p.name.startswith(prefix + "-")), None)
    if not matching:
        return False
    idx = blog_paths.index(matching)
    current = blog_title(matching)
    previous = blog_title(blog_paths[idx - 1]) if idx else None
    following = blog_title(blog_paths[idx + 1]) if idx + 1 < len(blog_paths) else None

    cells = nb.get("cells", [])
    if not any(c.get("cell_type") == "markdown" and "Where this lesson fits" in "".join(c.get("source", [])) for c in cells):
        cells.insert(1, {"cell_type": "markdown", "metadata": {}, "source": [bridge(prefix, current, previous, following) + "\n"]})

    anchor = exact_anchor(prefix)
    if anchor and not any(c.get("cell_type") == "markdown" and anchor[0].splitlines()[0] in "".join(c.get("source", [])) for c in cells):
        insert_at = next((i for i, c in enumerate(cells) if c.get("cell_type") == "code"), len(cells))
        cells.insert(insert_at, {"cell_type": "markdown", "metadata": {}, "source": [anchor[0] + "\n"]})
        cells.insert(insert_at + 1, {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [anchor[1] + "\n"]})

    # Make the notebook's first line point back to the authoritative blog.
    first = cells[0] if cells else None
    if first and first.get("cell_type") == "markdown":
        src = "".join(first.get("source", []))
        blog_link = f"https://github.com/manish7725/deeplearning/blob/main/blogs/{matching.name}"
        if blog_link not in src:
            src += f"\n\n📖 **Read the matching blog:** [{matching.name}]({blog_link})\n"
            first["source"] = src.splitlines(True)

    nb["cells"] = cells
    path.write_text(json.dumps(nb, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return True


def main() -> None:
    blog_paths = sorted(BLOGS.glob("*.md"))
    for path in blog_paths:
        patch_blog(path)
    for path in sorted(NOTEBOOKS.glob("*.ipynb")):
        patch_notebook(path, blog_paths)
    print(f"Student walkthrough fixes applied to {len(blog_paths)} blogs and {len(list(NOTEBOOKS.glob('*.ipynb')))} notebooks.")


if __name__ == "__main__":
    main()
