# xelatex -> bibtex -> xelatex x2.  Build log goes to build.log (NOT main.out: hyperref owns that name).
main.pdf: main.tex numbers.tex refs.bib $(wildcard sections/*.tex) $(wildcard tables/*.tex) $(wildcard figures/*.pdf)
	xelatex -interaction=nonstopmode -halt-on-error main.tex > build.log 2>&1 || (tail -30 build.log; exit 1)
	bibtex main >> build.log 2>&1 || (tail -30 build.log; exit 1)
	xelatex -interaction=nonstopmode -halt-on-error main.tex >> build.log 2>&1
	xelatex -interaction=nonstopmode -halt-on-error main.tex >> build.log 2>&1
	@grep -E "undefined|multiply defined|Overfull \\\\hbox \([0-9]{2,}" build.log | sort | uniq -c | head -20 || true

clean:
	rm -f main.aux main.bbl main.blg main.log main.out main.toc build.log

.PHONY: clean
