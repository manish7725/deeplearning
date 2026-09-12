from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from rebuild_notebooks_from_blogs import build

BLOGS = ROOT / 'blogs'
NOTEBOOKS = ROOT / 'notebooks'

# The course spine is lessons 01-37. Other numbered markdown files are kept
# untouched; they are outside the current 37-lesson notebook curriculum.
blogs = [p for p in BLOGS.glob('[0-9][0-9]-*.md') if 1 <= int(p.name[:2]) <= 37]
assert len(blogs) == 37, f'Expected exactly 37 course blogs, found {len(blogs)}'

for blog in sorted(blogs):
    out = NOTEBOOKS / f'{blog.stem}.ipynb'
    out.write_text(__import__('json').dumps(build(blog), indent=2, ensure_ascii=False) + '\n', encoding='utf-8')

print(f'Rebuilt {len(blogs)} notebooks from their corresponding blogs.')
