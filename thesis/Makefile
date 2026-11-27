# xelatex -> bibtex -> xelatex x2.  Build log goes to build.log (NOT main.out: hyperref owns that name).
# MKTEXTFM=0: fontspec probes for the template's Windows fonts; do not try to build TFM files for them.
TEX = MKTEXTFM=0 xelatex -interaction=nonstopmode -halt-on-error main.tex
main.pdf: main.tex hnuthesis.cls refs.bib $(wildcard chapters/*.tex) $(wildcard gen/*.tex) $(wildcard gen/tables/*.tex) $(wildcard gen/figures/*.pdf) $(wildcard figures/*)
	$(TEX) > build.log 2>&1 || (tail -30 build.log; exit 1)
	bibtex main >> build.log 2>&1 || (tail -30 build.log; exit 1)
	$(TEX) >> build.log 2>&1
	$(TEX) >> build.log 2>&1
	@grep -E "undefined|multiply defined|Overfull \\\\hbox \([0-9]{2,}" main.log | grep -v "Font shape" | sort | uniq -c | head -20 || true
	@# a number macro whose path has no value prints as ??path?? (see gen/numbers.tex)
	@if command -v pdftotext > /dev/null; then pdftotext main.pdf - | grep -o "??[^?]*??" | sort | uniq -c | head -20; fi || true

# gen/numbers.tex and gen/ob_numbers.tex define only the numbers that the text references.  After
# referencing a new one (or after new results), regenerate them; this needs the Python environment
# of the repository and results/.
PY ?= python
numbers:
	cd .. && MPLBACKEND=Agg $(PY) scripts/make_assets.py --out thesis/gen && MPLBACKEND=Agg $(PY) scripts/ob_assets.py

clean:
	rm -f main.aux main.bbl main.blg main.log main.out main.toc build.log

.PHONY: clean numbers
