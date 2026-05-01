"""Fix mermaid label quoting + display-mode matrices across all notes/*.md."""
from __future__ import annotations
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "notes"

# Chars that, if they appear in a mermaid node label, force us to quote it.
NEEDS_QUOTE = re.compile(r"<br|[×→←↑↓…−·∈≈≤≥≠ℝℕℤ°²³⁴⁵αβγδλμσφθΣΠ]|&[a-z]+;|/")

# Mermaid node:  ID[label]  ID(label)  ID([label])  ID{label}  ID(((label)))  etc.
# We only touch the simple [ ] ( ) { } variants; if label is already wrapped in
# double quotes we leave it alone.
NODE_RE = re.compile(
    r"(?P<pre>(?:^|[\s>|&-])[A-Za-z_][\w]*)"        # node id
    r"(?P<open>\[|\(\(\(|\(\(|\(|\{)"                # opening bracket
    r"(?P<label>(?!\")[^\[\]\(\)\{\}\n]+?)"          # label (not already quoted)
    r"(?P<close>\]|\)\)\)|\)\)|\)|\})"               # matching close
)

# Map open -> expected close so we only quote balanced shapes.
PAIR = {"[": "]", "(": ")", "((": "))", "(((": ")))", "{": "}"}


def quote_mermaid_labels(block: str) -> str:
    def repl(m: re.Match[str]) -> str:
        op, cl, lbl = m.group("open"), m.group("close"), m.group("label")
        if PAIR.get(op) != cl:
            return m.group(0)
        if not NEEDS_QUOTE.search(lbl):
            return m.group(0)
        # escape any embedded double quotes
        safe = lbl.replace('"', '#quot;').strip()
        return f'{m.group("pre")}{op}"{safe}"{cl}'

    return NODE_RE.sub(repl, block)


MERMAID_BLOCK_RE = re.compile(r"(```mermaid\n)(.*?)(\n```)", re.DOTALL)


def fix_mermaid(text: str) -> str:
    return MERMAID_BLOCK_RE.sub(
        lambda m: m.group(1) + quote_mermaid_labels(m.group(2)) + m.group(3),
        text,
    )


# Promote inline matrices to display mode.
# Matches a single-$ segment that contains \begin{*matrix}.
INLINE_MATRIX_RE = re.compile(
    r"(?<!\$)\$(?!\$)([^\n$]*?\\begin\{[pbvBV]?matrix\}[^\n$]*?\\end\{[pbvBV]?matrix\}[^\n$]*?)(?<!\$)\$(?!\$)"
)


def fix_matrices(text: str) -> str:
    # Replace each inline $ ... matrix ... $ with its own display block on a new
    # line. Surrounding text on that line is preserved.
    def repl(m: re.Match[str]) -> str:
        return f"\n\n$$\n{m.group(1).strip()}\n$$\n\n"

    return INLINE_MATRIX_RE.sub(repl, text)


def process(path: Path) -> bool:
    src = path.read_text(encoding="utf-8")
    out = fix_matrices(fix_mermaid(src))
    if out != src:
        path.write_text(out, encoding="utf-8")
        return True
    return False


def main() -> None:
    changed = []
    for md in sorted(ROOT.rglob("*.md")):
        if process(md):
            changed.append(md.relative_to(ROOT.parent).as_posix())
    print(f"Changed {len(changed)} files:")
    for f in changed:
        print(" ", f)


if __name__ == "__main__":
    main()
