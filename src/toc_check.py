import sys
import pymupdf
sys.path.insert(0, "src")
from ingest import looks_like_toc

doc = pymupdf.open(sys.argv[1])
for i, page in enumerate(doc, 1):
    if looks_like_toc(page.get_text()):
        print("Looks like a table of contents: page", i)