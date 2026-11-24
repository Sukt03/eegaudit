"""PDF helpers for the v2 build checks (run with the gptoss20b env python, which has PyMuPDF).
  python pdf_tools.py text  main.pdf out.txt      all pages, one '=== page N ===' header each
  python pdf_tools.py pages main.pdf              page count, page of the first and last reference entry and of the appendix
  python pdf_tools.py scan  main.pdf PATTERN      page and line of every regex hit
  python pdf_tools.py png   main.pdf OUTDIR p1 p2 ...   render pages (1-based) at 110 dpi
"""
import re, sys
import fitz

cmd, pdf = sys.argv[1], sys.argv[2]
doc = fitz.open(pdf)
if cmd == "text":
    with open(sys.argv[3], "w") as f:
        for i, p in enumerate(doc, 1):
            f.write(f"=== page {i} ===\n{p.get_text()}\n")
elif cmd == "pages":
    print("pages", len(doc))
    for i, p in enumerate(doc, 1):
        t = p.get_text()
        for key in ("References", "Technical Appendix", "Conclusion"):
            if re.search(r"^\s*" + key + r"\s*$", t, re.M):
                print(f"heading {key!r} on page {i}")
elif cmd == "scan":
    rx = re.compile(sys.argv[3])
    for i, p in enumerate(doc, 1):
        for j, line in enumerate(p.get_text().split("\n")):
            if rx.search(line):
                print(f"p{i} l{j}: {line[:160]}")
elif cmd == "png":
    out = sys.argv[3]
    for n in sys.argv[4:]:
        doc[int(n) - 1].get_pixmap(dpi=110).save(f"{out}/page_{int(n):02d}.png")
