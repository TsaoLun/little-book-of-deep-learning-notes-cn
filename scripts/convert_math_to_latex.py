#!/usr/bin/env python3
"""Convert Unicode math symbols and formulas to LaTeX format in markdown files."""

import re
import sys

# ---------------------------------------------------------------------------
# Character-level Unicode → LaTeX mapping
# ---------------------------------------------------------------------------
CHAR_MAP = {
    # Letterlike / Script symbols
    'ℒ': r'\mathcal{L}',
    'ℓ': r'\ell',
    'ℝ': r'\mathbb{R}',
    '𝒟': r'\mathcal{D}',
    'ŷ': r'\hat{y}',
    # Math operators
    '∇': r'\nabla',
    '∈': r'\in',
    '∅': r'\varnothing',
    '∘': r'\circ',
    '→': r'\to',
    '≥': r'\geq',
    '≤': r'\leq',
    '≠': r'\neq',
    '≃': r'\simeq',
    '∼': r'\sim',
    '≪': r'\ll',
    '√': r'\sqrt',
    '∀': r'\forall',
    '·': r'\cdot',
    '×': r'\times',
    '÷': r'\div',
    # Greek (commonly used in ML)
    'Σ': r'\sum',
    'Δ': r'\Delta',
    'η': r'\eta',
    'τ': r'\tau',
    'β': r'\beta',
    'γ': r'\gamma',
    'σ': r'\sigma',
    'π': r'\pi',
    'θ': r'\theta',
    'μ': r'\mu',
    'ψ': r'\psi',
    'ε': r'\varepsilon',
    'ϵ': r'\epsilon',
    'φ': r'\varphi',
    'ω': r'\omega',
    'ρ': r'\rho',
    'λ': r'\lambda',
    'α': r'\alpha',
    # Subscript digits
    '₀': '_{0}', '₁': '_{1}', '₂': '_{2}', '₃': '_{3}', '₄': '_{4}',
    '₅': '_{5}', '₆': '_{6}', '₇': '_{7}', '₈': '_{8}', '₉': '_{9}',
    # Subscript letters
    'ₙ': '_{n}', 'ₖ': '_{k}', 'ₓ': '_{x}', 'ₘ': '_{m}',
    'ₛ': '_{s}', 'ₜ': '_{t}', 'ₐ': '_{a}', 'ₑ': '_{e}', 'ₒ': '_{o}',
    'ᵢ': '_{i}', 'ⱼ': '_{j}',
    '₌': '=',
    # Superscript modifier letters (used for layer indices etc.)
    'ᴺ': '^{N}', 'ᴷ': '^{K}', 'ᴰ': '^{D}', 'ᴹ': '^{M}',
    'ᴬ': '^{A}', 'ᴮ': '^{B}', 'ᴵ': '^{I}', 'ᴶ': '^{J}',
    # Subscript-like modifier letters
    'ᵧ': '_{y}',
    # Superscript digits / operators (Unicode superscript range U+2070-U+2079 and Latin-1 ¹²³)
    '\u2070': '^{0}',
    '\u00b9': '^{1}', '\u00b2': '^{2}', '\u00b3': '^{3}',
    '\u2074': '^{4}', '\u2075': '^{5}', '\u2076': '^{6}',
    '\u2077': '^{7}', '\u2078': '^{8}', '\u2079': '^{9}',
    '\u207b': '^{-}',
    '\u207d': '^{(}', '\u207e': '^{)}',
    # En dash in math contexts → -
    '–': '-',
}

# Post-conversion regex fixups (applied to math content after char substitution)
_FIXUPS = [
    # \sum_{x}=_{y}^{Z} or \sum_{x}={y}^{Z}  →  \sum_{x=y}^{Z}
    (re.compile(r'\\sum_\{([^}]+)\}=_?\{?([^}^{ ]+)\}?\^\{([^}]+)\}'),
     r'\\sum_{\1=\2}^{\3}'),
    # \sum_{x}=_{y}  (no upper limit)
    (re.compile(r'\\sum_\{([^}]+)\}=_?\{?([^}^{ ]+)\}?(?!\^)'),
     r'\\sum_{\1=\2}'),
    # cleanup double braces: _{_{x}} → _{x}
    (re.compile(r'_\{_\{([^}]+)\}\}'), r'_{\1}'),
    (re.compile(r'\^\{_\{([^}]+)\}\}'), r'^{\1}'),
    # _{n}=_{1}^{N}  appearing outside \sum context (generic sub/sup sequence)
    (re.compile(r'_\{([a-zA-Z])\}=_?\{?([^}^{ ]+)\}?\^\{([^}]+)\}'),
     r'_{\1=\2}^{\3}'),
    # Merge consecutive superscripts: ^{a}^{b} → ^{ab}  (applied repeatedly)
    (re.compile(r'\^\{([^}]*)\}\^\{([^}]*)\}'), r'^{\1\2}'),
    (re.compile(r'\^\{([^}]*)\}\^\{([^}]*)\}'), r'^{\1\2}'),
    # \forall followed directly by a letter → add space
    (re.compile(r'\\forall([a-zA-Z])'), r'\\forall \1'),
    # Three consecutive \cdot → \cdots
    (re.compile(r'\\cdot\\cdot\\cdot'), r'\\cdots'),
]


def apply_fixups(s: str) -> str:
    for pattern, repl in _FIXUPS:
        s = pattern.sub(repl, s)
    return s

# Characters that signal "this text contains math"
MATH_TRIGGERS = set(CHAR_MAP.keys()) - {'–'}

# ---------------------------------------------------------------------------
# CJK detection
# ---------------------------------------------------------------------------
_CJK_PUNCT = set('，。、；：！？「」""（）【】《》…—')


def is_cjk(c: str) -> bool:
    if c in _CJK_PUNCT:
        return True
    cp = ord(c)
    return (
        0x4E00 <= cp <= 0x9FFF   # CJK Unified Ideographs
        or 0x3400 <= cp <= 0x4DBF  # CJK Extension A
        or 0x3000 <= cp <= 0x303F  # CJK Symbols and Punctuation
        or 0xFF01 <= cp <= 0xFF60  # Fullwidth forms
    )


def has_math(s: str) -> bool:
    return any(c in MATH_TRIGGERS for c in s)


def convert_chars(s: str) -> str:
    """Replace every unicode math character with its LaTeX equivalent."""
    converted = ''.join(CHAR_MAP.get(c, c) for c in s)
    return apply_fixups(converted)


# ---------------------------------------------------------------------------
# Equation-line detection
# ---------------------------------------------------------------------------

def _non_space_chars(s: str) -> int:
    return sum(1 for c in s if not c.isspace())


def is_pure_equation_line(line: str) -> bool:
    """True when the line is almost entirely math (standalone equation)."""
    stripped = line.strip()
    if not stripped:
        return False
    # Skip headers, code fences, already-LaTeX lines
    if stripped.startswith(('#', '```', '$', '-', '*', '>')):
        return False
    if not has_math(stripped):
        return False
    total = _non_space_chars(stripped)
    if total == 0:
        return False
    cjk_count = sum(1 for c in stripped if is_cjk(c))
    return cjk_count / total < 0.15


# ---------------------------------------------------------------------------
# Math-fragment finder for prose lines
# ---------------------------------------------------------------------------

def find_math_spans(line: str):
    """Return list of (start, end) index pairs of math expressions in *line*.

    Expands each MATH_TRIGGER character left/right until hitting a CJK char or
    line end, then merges overlapping spans.
    """
    if not has_math(line):
        return []

    n = len(line)
    raw_spans = []

    i = 0
    while i < n:
        if line[i] in MATH_TRIGGERS:
            # Expand left
            left = i
            while left > 0:
                prev = line[left - 1]
                if is_cjk(prev):
                    break
                if prev == '\n':
                    break
                left -= 1

            # Expand right
            right = i + 1
            while right < n:
                c = line[right]
                if is_cjk(c) or c == '\n':
                    break
                right += 1

            # Save scan end before trimming (always > i, safe for advance)
            scan_end = right

            # Trim trailing whitespace before CJK
            while right > left + 1 and line[right - 1] in ' \t':
                right -= 1
            # Trim trailing Chinese punctuation outside the math expression
            trimmed = line[left:right].rstrip()
            while trimmed and is_cjk(trimmed[-1]):
                trimmed = trimmed[:-1]
            right = left + len(trimmed)

            if right > left:
                raw_spans.append((left, right))
            # Always advance past what we scanned to avoid infinite loop
            i = scan_end
        else:
            i += 1

    if not raw_spans:
        return []

    # Merge overlapping / adjacent spans
    raw_spans.sort()
    merged = [list(raw_spans[0])]
    for s, e in raw_spans[1:]:
        if s <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])

    return [(s, e) for s, e in merged]


# ---------------------------------------------------------------------------
# Line processor
# ---------------------------------------------------------------------------

_LABEL_RE = re.compile(r'\s*\((\d+\.\d+)\)\s*$')


def wrap_equation(body: str) -> str:
    """Wrap an equation body in $$ ... $$, handling equation labels."""
    body = body.rstrip('。，、')
    m = _LABEL_RE.search(body)
    if m:
        label = m.group(1)
        eq = body[:m.start()].strip()
        return f'\n$$\n{eq} \\tag{{{label}}}\n$$\n'
    return f'\n$$\n{body.strip()}\n$$\n'


def process_prose_segment(seg: str) -> str:
    """Convert math fragments inside a non-LaTeX prose segment."""
    if not has_math(seg):
        return seg

    spans = find_math_spans(seg)
    if not spans:
        return seg

    result = []
    prev = 0
    for start, end in spans:
        result.append(seg[prev:start])
        fragment = seg[start:end]
        converted = convert_chars(fragment).strip().strip('.')
        if converted:
            result.append(f'${converted}$')
        prev = end
    result.append(seg[prev:])
    return ''.join(result)


def process_line(line: str) -> str:
    """Process a single markdown line (not inside a code block)."""
    stripped = line.strip()

    if not stripped:
        return line

    # Header lines: only convert math fragments within them
    if stripped.startswith('#'):
        if has_math(line):
            return process_line_with_existing_latex(line)
        return line

    # Already $$...$$ display math
    if stripped.startswith('$$'):
        return line

    # Standalone equation line → wrap in $$ ... $$
    if is_pure_equation_line(stripped):
        converted = convert_chars(stripped)
        return wrap_equation(converted)

    # Prose line: preserve existing $...$ and process the rest
    return process_line_with_existing_latex(line)


def process_line_with_existing_latex(line: str) -> str:
    """Split by existing $...$ blocks; only process non-LaTeX parts."""
    # Split on $...$ and $$...$$ blocks
    pattern = re.compile(r'(\$\$[\s\S]*?\$\$|\$[^$\n]+?\$)')
    parts = pattern.split(line)

    result = []
    for idx, part in enumerate(parts):
        if idx % 2 == 1:
            # Already-LaTeX block — preserve as-is
            result.append(part)
        else:
            # Plain text — process math fragments
            result.append(process_prose_segment(part))
    return ''.join(result)


# ---------------------------------------------------------------------------
# File processor
# ---------------------------------------------------------------------------

def process_file(path: str) -> None:
    with open(path, encoding='utf-8') as fh:
        content = fh.read()

    lines = content.split('\n')
    result = []
    in_code_block = False

    for line in lines:
        # Track fenced code blocks
        stripped = line.strip()
        if stripped.startswith('```'):
            in_code_block = not in_code_block
            result.append(line)
            continue

        if in_code_block:
            result.append(line)
            continue

        result.append(process_line(line))

    new_content = '\n'.join(result)

    if new_content != content:
        with open(path, 'w', encoding='utf-8') as fh:
            fh.write(new_content)
        print(f'Updated: {path}')
    else:
        print(f'No changes: {path}')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import glob
    import os

    targets = sys.argv[1:] if len(sys.argv) > 1 else sorted(
        glob.glob(os.path.join(os.path.dirname(__file__), '..', '[1-9]_*.md'))
    )
    for path in targets:
        process_file(path)
