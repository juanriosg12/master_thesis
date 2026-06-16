"""
ai_detector.py — AI-signal density analyzer for LaTeX thesis prose.

Usage:
    python ai_detector.py thesis_working_set/master_thesis.tex

Outputs:
    ai_detection_report.md   (per-section scores + flagged phrases with rewrites)

Scoring:
    density = sum(signal_points) / (word_count / 100)
    density 0-2  → score  0-25   Natural
    density 2-5  → score 26-50   Moderate
    density 5-8  → score 51-75   High
    density 8+   → score 76-100  Very High
"""

import re
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple


# ---------------------------------------------------------------------------
# Lexicon
# ---------------------------------------------------------------------------

class Signal(NamedTuple):
    pattern: str          # raw regex (case-insensitive)
    weight: int           # 1 low / 2 medium / 3 high
    rewrite_hint: str     # short suggestion shown in report


HIGH = 3
MED  = 2
LOW  = 1

LEXICON: list[Signal] = [
    # ----- HIGH weight (3 pts) -----
    Signal(r"\bit is worth noting\b",          HIGH, "Drop entirely — state the fact directly."),
    Signal(r"\bplays? (a|an|the) \w+ role\b",  HIGH, "Replace with a concrete verb: 'controls', 'determines', 'drives'."),
    Signal(r"\bin the context of\b",            HIGH, "Often deletable; or restructure the sentence."),
    Signal(r"\bthis (thesis|paper|work) addresses\b", HIGH, "Try 'We built / measured / tested…' instead."),
    Signal(r"\bopen gap\b",                     HIGH, "Say what the gap is; don't label it 'open'."),
    Signal(r"\bto mitigate (these|this|the)\b", HIGH, "Try 'To fix this…' or 'Because of this, we…'."),
    Signal(r"\bbuilding on this\b",             HIGH, "Cut the bridge; lead with the new point."),
    Signal(r"\bagainst this backdrop\b",        HIGH, "Cut the setup; start with the finding."),
    Signal(r"\bwith this in mind\b",            HIGH, "Cut — it adds no information."),
    Signal(r"\bparticularly important\b",       HIGH, "Delete 'particularly' and say why it matters instead."),
    Signal(r"\bcomprehensive experimental pipeline\b", HIGH, "Just 'experimental pipeline'."),
    Signal(r"\bcritical bottleneck\b",          HIGH, "Name the specific constraint instead."),
    Signal(r"\ba fundamental distinction must be\b", HIGH, "Try 'One distinction matters here:'."),
    Signal(r"\bserves as\b",                    HIGH, "Replace with a direct verb: 'is', 'acts as' → just describe what it does."),
    Signal(r"\bunderpins\b",                    HIGH, "Usually replaceable with 'drives', 'determines', or 'makes X possible'."),
    Signal(r"\bseamlessly\b",                   HIGH, "Delete — says nothing concrete."),

    # ----- MEDIUM weight (2 pts) -----
    Signal(r"\bleverages?\b",                   MED,  "'Uses', 'applies', or name the technique directly."),
    Signal(r"\brobust(ness)?\b",                MED,  "Say what specifically makes it robust, or what it's robust to."),
    Signal(r"\bnuanced\b",                      MED,  "Describe the nuance instead."),
    Signal(r"\bsystematically\b",               MED,  "Often redundant if the method is already described; delete or specify how."),
    Signal(r"\bcomprehensive\b",                MED,  "Delete — let the scope speak for itself."),
    Signal(r"\bfacilitates?\b",                 MED,  "'Allows', 'enables', or name the direct effect."),
    Signal(r"\bensures? that\b",                MED,  "Try 'so that' or restructure."),
    Signal(r"\benables?\b",                     MED,  "Often 'allows' is more direct; or name the specific effect."),
    Signal(r"\bfundamental(ly)?\b",             MED,  "Delete or replace with what makes it foundational."),
    Signal(r"\bdemonstrates?\b",                MED,  "'Shows', 'confirms', or 'proves' — pick the strongest that's accurate."),
    Signal(r"\bcritical(ly)?\b",                MED,  "Overused. Only keep if you mean 'the experiment fails without this'."),
    Signal(r"\baddresses?\b",                   MED,  "Prefer concrete verbs: 'fixes', 'removes', 'resolves'."),
    Signal(r"\bkey\b",                          MED,  "Delete — everything in a thesis is 'key'."),
    Signal(r"\bpivotal\b",                      MED,  "Same as 'key' — delete or explain why."),
    Signal(r"\bcrucial(ly)?\b",                 MED,  "Same as 'critical' — overused. Justify or delete."),
    Signal(r"\bimportant(ly)?\b",               MED,  "Say WHY it is important instead of asserting it."),
    Signal(r"\bhighlights?\b",                  MED,  "'Shows', 'confirms', 'points to'."),
    Signal(r"\bshed(s)? light\b",               MED,  "'Explains', 'reveals', 'shows'."),

    # ----- LOW weight (1 pt) -----
    Signal(r"\bdelves?\b",                      LOW,  "'Examines', 'analyzes', 'looks at' — 'delve' is AI-favored."),
    Signal(r"\breveals?\b",                     LOW,  "Often fine; flag if the sentence could just state the finding directly."),
    Signal(r"\baligns? with\b",                 LOW,  "Often 'matches', 'is consistent with', or just delete."),
    Signal(r"\bexplores?\b",                    LOW,  "Name what you actually did instead."),
    Signal(r"\bstands? out\b",                  LOW,  "Fine colloquially; flag if overused in formal sections."),
    Signal(r"\bemerge[sd]?\b",                  LOW,  "'Appear', 'arise', 'we find' — watch for AI-favored usage."),
    Signal(r"\binteresting(ly)?\b",             LOW,  "Delete — show what's interesting instead."),
    Signal(r"\bnotably\b",                      LOW,  "Often fine; flag when used as sentence-opener to pad a finding."),
    Signal(r"\bit should be noted\b",           LOW,  "State the note directly."),
    Signal(r"\bcan be seen\b",                  LOW,  "Replace passive: 'X shows…', 'the figure shows…'."),
    Signal(r"\bone can observe\b",              LOW,  "Just state the observation."),
    Signal(r"\bthis suggests? that\b",          LOW,  "Just make the claim: 'X therefore Y'."),
]


# ---------------------------------------------------------------------------
# LaTeX stripping
# ---------------------------------------------------------------------------

ENV_PATTERN = re.compile(
    r'\\begin\{('
    r'figure|table|algorithm|algorithmic|align|equation|tabular|tabularx'
    r'|minipage|subfigure|lstlisting|verbatim|itemize|enumerate|description'
    r')\*?\}.*?\\end\{\1\*?\}',
    re.DOTALL | re.IGNORECASE,
)

DISPLAY_MATH = re.compile(r'\\\[.*?\\\]', re.DOTALL)
INLINE_MATH  = re.compile(r'\$\$.*?\$\$', re.DOTALL)
SINGLE_MATH  = re.compile(r'\$[^$\n]{1,120}\$')
COMMAND      = re.compile(r'\\[a-zA-Z]+\*?(?:\[[^\]]*\])*(?:\{[^}]*\})*')
COMMENT      = re.compile(r'%.*')
LABEL_CMD    = re.compile(r'\\(label|ref|cite[tp]?|eqref|autoref)\{[^}]*\}')
SECTION_CMD  = re.compile(
    r'\\(section|subsection|subsubsection|paragraph|subparagraph)\*?\{([^}]+)\}'
)


def section_title(raw: str) -> str:
    """Strip any remaining LaTeX from a section title string."""
    t = COMMAND.sub('', raw)
    t = re.sub(r'[{}]', '', t)
    return t.strip()


def strip_latex(text: str) -> str:
    """Remove LaTeX markup, leaving plain prose words."""
    text = COMMENT.sub('', text)
    text = ENV_PATTERN.sub('', text)
    text = DISPLAY_MATH.sub('', text)
    text = INLINE_MATH.sub('', text)
    text = SINGLE_MATH.sub(' MATHEXPR ', text)
    text = LABEL_CMD.sub('', text)
    # keep section titles as plain text
    text = SECTION_CMD.sub(lambda m: m.group(2), text)
    text = COMMAND.sub(' ', text)
    text = re.sub(r'[{}]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# ---------------------------------------------------------------------------
# Section splitting (preserves line numbers)
# ---------------------------------------------------------------------------

@dataclass
class SectionBlock:
    name: str
    lines: list[tuple[int, str]] = field(default_factory=list)  # (1-based line no, raw line)


SECTION_RE = re.compile(
    r'^\s*\\(section|subsection|subsubsection)\*?\{([^}]+)\}'
)
APPENDIX_RE = re.compile(r'^\s*\\appendix\b')


def split_sections(raw_lines: list[str]) -> list[SectionBlock]:
    blocks: list[SectionBlock] = [SectionBlock("Preamble")]
    in_appendix = False

    for i, line in enumerate(raw_lines, start=1):
        if APPENDIX_RE.match(line):
            in_appendix = True
            blocks.append(SectionBlock("Appendix A"))
            continue

        m = SECTION_RE.match(line)
        if m:
            level, title = m.group(1), m.group(2)
            clean = section_title(title)
            prefix = {"section": "§", "subsection": "  §", "subsubsection": "    §"}[level]
            full_name = f"{prefix} {clean}"
            if in_appendix:
                full_name = f"Appendix — {clean}"
            # top-level sections start a new block; subsections stay inside
            if level == "section":
                blocks.append(SectionBlock(full_name))
            else:
                # Append a sub-header note but keep in current top block
                blocks[-1].lines.append((i, line))
                continue

        blocks[-1].lines.append((i, line))

    return [b for b in blocks if b.lines]


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

@dataclass
class Match:
    lineno: int
    phrase: str
    context: str      # surrounding sentence
    weight: int
    rewrite_hint: str


def density_to_score(density: float) -> tuple[int, str]:
    if density <= 2.0:
        return int(density / 2.0 * 25), "Natural — minimal rework needed"
    if density <= 5.0:
        return 25 + int((density - 2.0) / 3.0 * 25), "Moderate — some rework recommended"
    if density <= 8.0:
        return 50 + int((density - 5.0) / 3.0 * 25), "High — significant rework needed"
    capped = min(density, 15.0)
    return 75 + int((capped - 8.0) / 7.0 * 25), "Very High — heavy rework"


def find_context(plain: str, match_start: int, match_end: int, window: int = 120) -> str:
    """Return the sentence window around a match."""
    start = max(0, match_start - window)
    end   = min(len(plain), match_end + window)
    snippet = plain[start:end].strip()
    # trim to nearest sentence boundary
    if start > 0 and '.' in plain[start:match_start]:
        idx = plain[start:match_start].rfind('.')
        snippet = plain[start + idx + 1:end].strip()
    return re.sub(r'\s+', ' ', snippet)


def analyze_section(block: SectionBlock) -> tuple[list[Match], int, float, int, str]:
    """Returns (matches, total_points, density, score, label)."""
    raw_text = '\n'.join(line for _, line in block.lines)
    plain    = strip_latex(raw_text)
    words    = len(plain.split())

    if words < 30:
        return [], 0, 0.0, 0, "N/A (too short)"

    matches: list[Match] = []
    total_pts = 0

    # Build a line-number lookup: character offset → line number
    # We work on the raw lines to get accurate line numbers
    line_offsets: list[tuple[int, int]] = []  # (char_start, lineno)
    offset = 0
    for lineno, line in block.lines:
        line_offsets.append((offset, lineno))
        offset += len(line) + 1  # +1 for newline

    def char_to_lineno(char_idx: int) -> int:
        lo, hi = 0, len(line_offsets) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if line_offsets[mid][0] <= char_idx:
                lo = mid
            else:
                hi = mid - 1
        return line_offsets[lo][1]

    for sig in LEXICON:
        for m in re.finditer(sig.pattern, plain, re.IGNORECASE):
            phrase   = m.group(0)
            ctx      = find_context(plain, m.start(), m.end())
            # approximate line number: find phrase in raw_text
            raw_pos  = raw_text.lower().find(phrase.lower())
            lineno   = char_to_lineno(max(0, raw_pos)) if raw_pos >= 0 else block.lines[0][0]
            matches.append(Match(
                lineno=lineno,
                phrase=phrase,
                context=ctx,
                weight=sig.weight,
                rewrite_hint=sig.rewrite_hint,
            ))
            total_pts += sig.weight

    density = total_pts / (words / 100) if words > 0 else 0
    score, label = density_to_score(density)
    return matches, total_pts, density, score, label


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------

WEIGHT_LABEL = {HIGH: "⚠ HIGH", MED: "◆ MED", LOW: "◇ low"}
SCORE_BARS   = {
    (0,  26): "██░░░░░░░░",
    (26, 51): "████░░░░░░",
    (51, 76): "██████░░░░",
    (76, 101):"██████████",
}

def score_bar(score: int) -> str:
    for (lo, hi), bar in SCORE_BARS.items():
        if lo <= score < hi:
            return bar
    return "██████████"


def write_report(sections: list[SectionBlock], path: Path) -> None:
    results = []
    for block in sections:
        if block.name == "Preamble":
            continue
        matches, pts, density, score, label = analyze_section(block)
        results.append((block.name, matches, pts, density, score, label))

    # Overall (word-count weighted)
    total_pts_all = sum(r[2] for r in results)
    total_words_all = sum(
        len(strip_latex('\n'.join(l for _, l in b.lines)).split())
        for b in sections if b.name != "Preamble"
    )
    overall_density = total_pts_all / (total_words_all / 100) if total_words_all else 0
    overall_score, overall_label = density_to_score(overall_density)

    lines = []
    lines.append("# AI-Signal Detection Report — `master_thesis.tex`\n")
    lines.append(f"> Generated by `ai_detector.py`\n")
    lines.append("---\n")
    lines.append("## Overall Score\n")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| **Overall AI-signal score** | **{overall_score}/100** |")
    lines.append(f"| Assessment | {overall_label} |")
    lines.append(f"| Signal density | {overall_density:.2f} pts / 100 words |")
    lines.append(f"| Total prose words | {total_words_all:,} |")
    lines.append(f"| Total signal points | {total_pts_all} |\n")
    lines.append(f"{score_bar(overall_score)} `{overall_score}/100`\n")
    lines.append("---\n")
    lines.append("## Per-Section Breakdown\n")
    lines.append("| Section | Score | Density | Assessment | Flags |")
    lines.append("|---------|-------|---------|------------|-------|")

    for name, matches, pts, density, score, label in results:
        n_flags = len(matches)
        bar = score_bar(score)
        lines.append(
            f"| {name} | `{score}/100` {bar} | {density:.1f} | {label} | {n_flags} |"
        )

    lines.append("\n---\n")
    lines.append("## Flagged Phrases by Section\n")
    lines.append(
        "> Format: **Phrase** · weight · line number  \n"
        "> _Context_ (surrounding sentence)  \n"
        "> 💡 Suggested rewrite direction\n"
    )

    for name, matches, pts, density, score, label in results:
        if not matches:
            continue
        lines.append(f"\n### {name.strip()}  `{score}/100`\n")
        # sort by weight descending, then line number
        for m in sorted(matches, key=lambda x: (-x.weight, x.lineno)):
            wlabel = WEIGHT_LABEL[m.weight]
            ctx_wrapped = textwrap.fill(f'"{m.context}"', width=90, subsequent_indent="  ")
            lines.append(f"- **`{m.phrase}`** · {wlabel} · line {m.lineno}")
            lines.append(f"  > {ctx_wrapped}")
            lines.append(f"  💡 {m.rewrite_hint}\n")

    lines.append("---\n")
    lines.append("## Scoring Guide\n")
    lines.append("| Density range | Score | What it means |")
    lines.append("|---|---|---|")
    lines.append("| 0 – 2 | 0–25 | **Natural** — prose reads direct and personal |")
    lines.append("| 2 – 5 | 26–50 | **Moderate** — a few hedges/fillers; targeted cleanup |")
    lines.append("| 5 – 8 | 51–75 | **High** — systematic AI patterns; section needs rework |")
    lines.append("| 8+ | 76–100 | **Very High** — heavy rework recommended |")
    lines.append("\n> Score measures **rewriting priority**, not a binary human/AI classifier.")
    lines.append("> Captions, algorithms, tables, and math are excluded from scoring.")

    path.write_text('\n'.join(lines), encoding='utf-8')
    print(f"Report written → {path}")
    print(f"Overall score: {overall_score}/100 ({overall_label})")
    print(f"Density: {overall_density:.2f} pts / 100 words")
    print()
    print("Per-section scores:")
    for name, _, _, density, score, label in results:
        print(f"  {score:3d}/100  {density:4.1f}  {name.strip()}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python ai_detector.py <path/to/master_thesis.tex>")
        sys.exit(1)

    tex_path = Path(sys.argv[1])
    if not tex_path.exists():
        print(f"File not found: {tex_path}")
        sys.exit(1)

    raw_lines = tex_path.read_text(encoding='utf-8', errors='replace').splitlines()
    sections  = split_sections(raw_lines)
    out_path  = Path("ai_detection_report.md")
    write_report(sections, out_path)


if __name__ == "__main__":
    main()
