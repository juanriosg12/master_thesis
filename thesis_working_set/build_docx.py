#!/usr/bin/env python3
"""
Build Master_Thesis_Causal_Shapley_v7.docx from the markdown source.

WHY THIS SCRIPT EXISTS
----------------------
Plain `pandoc file.md -o file.docx` produces a Word file whose TABLES render as
broken empty grids (cell text spills outside the grid) in LibreOffice / some
Word configs. Two root causes, both handled below:

  1. Pandoc tags every table with `tblStyle="Table"` and every cell paragraph
     with `pStyle="Compact"`, but neither style is defined in the reference doc,
     so the renderer builds an empty grid skeleton and drops the text outside it.
     FIX: strip both style references.
  2. Pandoc tables have `tblW type="auto" w=0`, no `tblLayout`, no per-cell
     `tcW`, and no borders -> columns collapse to zero width.
     FIX: set fixed tblW (= sum of gridCols), add `tblLayout=fixed`, inject
     per-cell `tcW` + `tcMar`, and add visible `tblBorders`.

Also: pandoc omits the PNG content-type declaration -> add it so images validate.

PREREQUISITES (already present in this environment)
---------------------------------------------------
  - pandoc 3.x
  - the markdown file + the figures/ folder in the SAME directory as this script
  - reference.docx (regenerated below if missing) for fonts/margins/page size
  - the docx skill scripts at /mnt/skills/public/docx/scripts/office/

FORMAT BAKED IN
---------------
  Times New Roman 12, single spacing, US Letter (8.5x11), 1" margins.
  Headings: H1 15pt bold, H2 13pt bold, H3 12pt bold (all Times New Roman).

USAGE
-----
  python build_docx.py
  # -> Master_Thesis_Causal_Shapley_v7.docx
"""
import re, subprocess, os, sys

SRC = "Master_Thesis_Causal_Shapley_v7.md"
OUT = "Master_Thesis_Causal_Shapley_v7.docx"
REF = "reference.docx"
OFFICE = "/mnt/skills/public/docx/scripts/office"

# --- 0. regenerate reference.docx (styles) if absent -------------------------
def make_reference():
    js = r'''
const { Document, Packer, Paragraph } = require("docx");
const fs = require("fs");
const tnr = "Times New Roman";
const doc = new Document({
  styles: {
    default: { document: { run: { font: tnr, size: 24 },
      paragraph: { spacing: { line: 240, lineRule: "auto", after: 120 } } } },
    paragraphStyles: [
      { id: "Title", name: "Title", basedOn: "Normal", next: "Normal",
        run: { font: tnr, size: 32, bold: true }, paragraph: { spacing: { after: 240 } } },
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: tnr, size: 30, bold: true },
        paragraph: { spacing: { before: 240, after: 120 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: tnr, size: 26, bold: true },
        paragraph: { spacing: { before: 200, after: 100 }, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: tnr, size: 24, bold: true },
        paragraph: { spacing: { before: 160, after: 80 }, outlineLevel: 2 } },
    ]
  },
  sections: [{ properties: { page: {
    size: { width: 12240, height: 15840 },
    margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } } },
    children: [ new Paragraph({ text: "ref" }) ] }]
});
Packer.toBuffer(doc).then(b => fs.writeFileSync("reference.docx", b));
'''
    open("_mkref.js", "w").write(js)
    subprocess.run(["node", "_mkref.js"], check=True)
    os.remove("_mkref.js")

if not os.path.exists(REF):
    make_reference()

# --- 1. pandoc convert (pipe_tables extension is REQUIRED) -------------------
subprocess.run(["pandoc", SRC, "-f", "markdown+pipe_tables",
                f"--reference-doc={REF}", "--resource-path=.", "-o", "_raw.docx"], check=True)

# --- 2. unpack ---------------------------------------------------------------
subprocess.run(["python", f"{OFFICE}/unpack.py", "_raw.docx", "unpacked/"],
               check=True, stdout=subprocess.DEVNULL)

p = "unpacked/word/document.xml"
x = open(p).read()

# --- 3. strip undefined style refs that break table rendering ----------------
x = x.replace('<w:tblStyle w:val="Table"/>', '')
x = x.replace('<w:pStyle w:val="Compact"/>', '')

# --- 4. fix every table: fixed width, layout, borders, per-cell tcW + margins -
BORDERS = ('<w:tblBorders>'
 '<w:top w:val="single" w:sz="4" w:space="0" w:color="999999"/>'
 '<w:left w:val="single" w:sz="4" w:space="0" w:color="999999"/>'
 '<w:bottom w:val="single" w:sz="4" w:space="0" w:color="999999"/>'
 '<w:right w:val="single" w:sz="4" w:space="0" w:color="999999"/>'
 '<w:insideH w:val="single" w:sz="4" w:space="0" w:color="999999"/>'
 '<w:insideV w:val="single" w:sz="4" w:space="0" w:color="999999"/></w:tblBorders>')
CELLMAR = ('<w:tcMar><w:top w:w="40" w:type="dxa"/><w:left w:w="80" w:type="dxa"/>'
           '<w:bottom w:w="40" w:type="dxa"/><w:right w:w="80" w:type="dxa"/></w:tcMar>')

def fix_table(m):
    tbl = m.group(0)
    cols = [int(w) for w in re.findall(r'<w:gridCol w:w="(\d+)"', tbl)]
    if not cols:
        return tbl
    total = sum(cols)
    tbl = re.sub(r'<w:tblW[^/]*/>', f'<w:tblW w:type="dxa" w:w="{total}"/>', tbl)
    # tblBorders + tblLayout must sit in the right schema slot (after jc, or after tblW)
    if '<w:jc ' in tbl.split('</w:tblPr>')[0]:
        tbl = re.sub(r'(<w:tblPr>.*?<w:jc[^/]*/>)',
                     r'\1' + BORDERS + '<w:tblLayout w:type="fixed"/>',
                     tbl, count=1, flags=re.DOTALL)
    else:
        tbl = re.sub(r'(<w:tblW[^/]*/>)',
                     r'\1' + BORDERS + '<w:tblLayout w:type="fixed"/>',
                     tbl, count=1)
    def fix_row(rm):
        row = rm.group(0); ci = [0]
        def fix_cell(cm):
            cell = cm.group(0); idx = ci[0]; ci[0] += 1
            w = cols[idx] if idx < len(cols) else cols[-1]
            return re.sub(r'<w:tcPr\s*/>',
                          f'<w:tcPr><w:tcW w:type="dxa" w:w="{w}"/>{CELLMAR}</w:tcPr>',
                          cell, count=1)
        return re.sub(r'<w:tc>.*?</w:tc>', fix_cell, row, flags=re.DOTALL)
    return re.sub(r'<w:tr>.*?</w:tr>', fix_row, tbl, flags=re.DOTALL)

x = re.sub(r'<w:tbl>.*?</w:tbl>', fix_table, x, flags=re.DOTALL)
open(p, "w").write(x)

# --- 5. declare PNG content type ---------------------------------------------
ct = "unpacked/[Content_Types].xml"
c = open(ct).read()
if 'Extension="png"' not in c:
    c = c.replace('<Override', '<Default Extension="png" ContentType="image/png"/><Override', 1)
    open(ct, "w").write(c)

print("tables fixed:", x.count('<w:tblBorders>'))

# --- 6. repack (validates) ---------------------------------------------------
subprocess.run(["python", f"{OFFICE}/pack.py", "unpacked/", OUT, "--original", "_raw.docx"], check=True)
print("built:", OUT)
