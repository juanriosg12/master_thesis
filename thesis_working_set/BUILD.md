# Build Instructions — Master Thesis LaTeX

## Files

| File | Description |
|---|---|
| `master_thesis.tex` | Main LaTeX source (compile this) |
| `references.bib` | BibTeX bibliography |
| `figures/` | All figures (PNG) already referenced in the `.tex` |

## Compile (4-step standard sequence)

```bash
cd /path/to/thesis_working_set

pdflatex master_thesis.tex   # first pass – builds .aux
bibtex master_thesis         # resolves citations
pdflatex master_thesis.tex   # second pass – inserts references
pdflatex master_thesis.tex   # third pass – fixes cross-refs & TOC
```

Output: `master_thesis.pdf`

## Installing LaTeX on macOS

### Option A – MacTeX (full, ~4 GB, recommended)
```bash
brew install --cask mactex-no-gui   # command-line tools only
# or
brew install --cask mactex          # includes GUI apps (TeXShop, etc.)
```
After install, restart the terminal so `/Library/TeX/texbin` is on your PATH.

### Option B – BasicTeX (minimal, ~100 MB) + add packages manually
```bash
brew install --cask basictex
sudo tlmgr update --self
sudo tlmgr install \
    algorithm2e algorithmicx \
    booktabs tabularx multirow \
    caption subcaption \
    setspace microtype \
    natbib
```

## Compile with a single Make command (optional)

```makefile
# Makefile
MAIN = master_thesis
.PHONY: pdf clean

pdf:
	pdflatex $(MAIN)
	bibtex   $(MAIN)
	pdflatex $(MAIN)
	pdflatex $(MAIN)

clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.toc *.lof *.lot
```

Run: `make pdf`

## Compile online (no local install needed)

Upload `master_thesis.tex`, `references.bib`, and the `figures/` folder to
[Overleaf](https://www.overleaf.com/). Set the compiler to **pdfLaTeX** and press
"Recompile". Overleaf handles the multi-pass compilation automatically.

## Required LaTeX packages

All packages below are included in a standard MacTeX / TexLive / MiKTeX installation:

```
lmodern, microtype, geometry, setspace,
amsmath, amssymb, mathtools, bm,
algorithm, algpseudocode,
booktabs, tabularx, multirow, array,
graphicx, float, subcaption, caption,
xcolor, hyperref, natbib,
enumitem, parskip, fancyhdr
```
