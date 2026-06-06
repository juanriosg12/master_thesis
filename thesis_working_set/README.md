# Thesis working set

## Files
- `Master_Thesis_Causal_Shapley_v7.md` — the canonical document. **Edit this.**
- `figures/` — all 27 figures (plus a few unused instance-TGA plots). Keep this
  folder next to the `.md`; figures are referenced as `figures/xxx.png`.
- `build_docx.py` — one-command converter to a properly formatted Word file.

## Iterate fast (markdown)
Edit the `.md` directly. To preview with images, open the folder in VS Code and
press Cmd/Ctrl+Shift+V (the `figures/` folder must sit beside the `.md`).

## Produce the Word file
From this folder (with pandoc + the docx skill scripts available):

    python build_docx.py

Output: `Master_Thesis_Causal_Shapley_v7.docx` — Times New Roman 12, single
spacing, US Letter, all figures embedded, all tables as real bordered grids.

The script exists because a plain `pandoc md -o docx` renders the tables as
broken empty grids. `build_docx.py` post-processes the table XML (fixed widths,
borders, cell margins, removes undefined pandoc styles) and declares the PNG
content type. See the docstring in the script for the full rationale.

## Conventions to keep (so conversion stays painless)
- **Tables**: standard markdown pipe tables, with a blank line before the header
  row and after the last row. Keep column counts consistent within a table.
- **Table captions**: `***Table N.M: Title***` on its own line, blank line above
  and below. Do NOT let stray `*` runs creep into a caption (an earlier
  `X-******>******X` artifact broke conversion until cleaned to `X->X`).
- **Figures**: `![alt](figures/name.png)` then a blank line then the caption
  `***Figure N.M: Title.*** body...`. Figures are numbered 4.1–4.27, contiguous;
  in-text references use `(Figure N.M)`. If you add/remove a figure, renumber so
  captions stay unique and contiguous and update any `(Figure N.M)` cross-refs.
- **Numbering today**: Tables 3.1–3.4 and 4.1–4.6; Figures 4.1–4.27.
- **Algorithm pseudocode**: fenced ``` blocks. Helper subroutines live in
  Appendix A (A.1–A.4); the main 3.4 blocks call them by name.
