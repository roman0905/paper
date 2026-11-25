# xelatex -> bibtex -> xelatex x2.  Build log goes to build.log (NOT main.out: hyperref owns that name).
# MKTEXTFM=0: fontspec probes for the template's Windows fonts; do not try to build TFM files for them.
TEX = MKTEXTFM=0 xelatex -interaction=nonstopmode -halt-on-error main.tex
main.pdf: main.tex hnuthesis.cls refs.bib $(wildcard chapters/*.tex) $(wildcard gen/*.tex) $(wildcard gen/tables/*.tex) $(wildcard gen/figures/*.pdf) $(wildcard figures/*)
	$(TEX) > build.log 2>&1 || (tail -30 build.log; exit 1)
	bibtex main >> build.log 2>&1 || (tail -30 build.log; exit 1)
	$(TEX) >> build.log 2>&1
	$(TEX) >> build.log 2>&1
	@grep -E "undefined|multiply defined|Overfull \\\\hbox \([0-9]{2,}" main.log | sort | uniq -c | head -20 || true

clean:
	rm -f main.aux main.bbl main.blg main.log main.out main.toc build.log

.PHONY: clean
