import sys
import pymupdf

doc = pymupdf.open(sys.argv[1])
page = doc[int(sys.argv[2]) - 1]  # page number as printed, starting at 1
print(page.get_text())
