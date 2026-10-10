"""Read the rulebook PDFs in data/pdfs and store embedded chunks in Chroma.

Run with: uv run python src/ingest.py
"""

import collections
import pathlib
import re
import shutil
import textwrap

import chromadb
import ollama
import pymupdf

PDF_DIR = pathlib.Path("data/pdfs")
DB_DIR = pathlib.Path("data/chroma")
COLLECTION = "rules"
EMBED_MODEL = "nomic-embed-text"

# Chunks aim for 300-500 characters. Only the ceiling can be enforced: a short
# numbered step or the tail of a section is as long as it is.
CHUNK_MAX = 500
EMBED_BATCH = 64

# A line counts as a heading if it is this much bigger than the body text.
HEADING_RATIO = 1.15
HEADING_MAX_CHARS = 60


# ---------------------------------------------------------------- page layout

def body_size(doc):
    """The font size most of the document is set in, used as the heading baseline."""
    sizes = collections.Counter()
    for page in doc:
        for block in page.get_text("rawdict")["blocks"]:
            if block["type"] != 0:
                continue
            for line in block["lines"]:
                for span in line["spans"]:
                    text = span_text(span).strip()
                    if text:
                        sizes[round(span["size"], 1)] += len(text)
    return sizes.most_common(1)[0][0] if sizes else 10.0


def reading_order(page):
    """Text blocks of a page in the order a human reads them.

    PyMuPDF hands back blocks in PDF content order, which on these two-column
    rulebooks puts titles next to each other and strands paragraphs under the
    wrong heading. Full-width blocks (titles, banners) split the page into
    bands; inside a band we read the left column top to bottom, then the right.
    """
    blocks = [b for b in page.get_text("rawdict")["blocks"] if b["type"] == 0 and b["lines"]]
    if not blocks:
        return []

    left = min(b["bbox"][0] for b in blocks)
    right = max(b["bbox"][2] for b in blocks)
    middle = (left + right) / 2
    tolerance = (right - left) * 0.05

    ordered, band = [], []

    def flush_band():
        # False (left column) sorts before True (right column).
        band.sort(key=lambda b: ((b["bbox"][0] + b["bbox"][2]) / 2 > middle, b["bbox"][1]))
        ordered.extend(band)
        band.clear()

    for block in sorted(blocks, key=lambda b: b["bbox"][1]):
        spans_both_columns = block["bbox"][0] < middle - tolerance and block["bbox"][2] > middle + tolerance
        if spans_both_columns:
            flush_band()
            ordered.append(block)
        else:
            band.append(block)
    flush_band()
    return ordered


# ----------------------------------------------------------------- text bits

def span_text(span):
    return "".join(c["c"] for c in span["chars"])


def line_text(line):
    return " ".join("".join(span_text(s) for s in line["spans"]).split())


def line_size(line):
    return max(span["size"] for span in line["spans"])


def is_letter_spaced(text):
    """True for titles typeset as "G A M E  OV E RV I E W"."""
    tokens = text.split()
    return len(tokens) >= 4 and sum(len(t) == 1 for t in tokens) / len(tokens) >= 0.5


def unspaced_line(line):
    """Rebuild a letter-spaced title into real words.

    The tracking is made of real space glyphs, so the text alone cannot tell a
    letter break from a word break. The geometry can: a word break leaves a
    visible gap before the space, a letter break does not.
    """
    size = line_size(line)
    out, previous_right = [], None
    for span in line["spans"]:
        for char in span["chars"]:
            gap = char["bbox"][0] - previous_right if previous_right is not None else 0.0
            if char["c"] != " ":
                out.append(char["c"])
            elif gap > size * 0.1:
                out.append(" ")
            previous_right = char["bbox"][2]
    return " ".join("".join(out).split())


def clean(text):
    """Drop table-of-contents dot leaders and other printing artefacts."""
    return " ".join(re.sub(r"\.{4,}", " ", text).split())


def wraps(previous, line):
    """True when `line` continues `previous` rather than starting a new title.

    A wrapped title (a long FAQ question) sits a fraction of its font size
    below the line above. Two separate titles that PyMuPDF has lumped into one
    block sit much further apart, and often come out in reverse order, which
    makes the gap negative.
    """
    gap = line["bbox"][1] - previous["bbox"][3]
    return 0 <= gap <= line_size(line) * 0.6


def looks_like_heading(text):
    """Reject oversized text that is really a sentence, not a title.

    Some rulebooks set callouts a size or two above the body text, so size
    alone is not enough: titles are short, start a phrase, and do not run on
    past a full stop.
    """
    return (
        0 < len(text) <= HEADING_MAX_CHARS
        and any(c.isalpha() for c in text)
        and not text.endswith((".", ",", ";", ":"))
        and not text[0].islower()
        and not re.search(r"\.\s", text)
    )


def is_flavour(block, source):
    """True for the italic story text that sits between the actual rules.

    In the rulebooks italic means flavour: in-world quotes, sidebars, the poem
    in Everdell. Worked examples are also italic but are rules content, so they
    stay. The FAQ uses italic to quote card text, so it is left alone entirely.
    """
    if source != "rules":
        return False
    spans = [s for line in block["lines"] for s in line["spans"] if span_text(s).strip()]
    if not spans or not all(span["flags"] & 2 for span in spans):  # flag 2 == italic
        return False
    text = " ".join(line_text(line) for line in block["lines"]).lstrip()
    return not text.startswith("Example")


# -------------------------------------------------------------- section split

def read_sections(path, source):
    """Yield (section, paragraphs) for one PDF.

    paragraphs is a list of (page, text) so every chunk can keep its page.
    """
    doc = pymupdf.open(path)
    baseline = body_size(doc) * HEADING_RATIO

    section, section_page, paragraphs = "", 1, []

    def finish():
        # A title with no body under it is still real text - component lists and
        # diagram labels are set large and read as titles - so keep it as a
        # one-line chunk instead of dropping it on the floor.
        if paragraphs or section:
            return section, paragraphs or [(section_page, section)]
        return None

    for number, page in enumerate(doc, start=1):
        for block in reading_order(page):
            if is_flavour(block, source):
                continue
            for heading, text in split_block(block, baseline):
                if heading:
                    done = finish()
                    if done:
                        yield done
                    section, section_page, paragraphs = text, number, []
                elif text:
                    paragraphs.append((number, text))
    done = finish()
    if done:
        yield done
    doc.close()


def split_block(block, baseline):
    """Walk a block's lines, yielding (is_heading, text) pieces.

    Runs of oversized lines are merged first, so a wrapped FAQ question becomes
    one heading. A whole paragraph that happens to be set large then merges into
    something too long to pass for a heading, which keeps it out of the titles.
    """
    lines = block["lines"]
    body, index = [], 0
    while index < len(lines):
        if line_size(lines[index]) >= baseline:
            start = index
            index += 1
            while (
                index < len(lines)
                and line_size(lines[index]) >= baseline
                and wraps(lines[index - 1], lines[index])
            ):
                index += 1
            run = lines[start:index]
            text = clean(" ".join(line_text(line) for line in run))
            if looks_like_heading(text):
                if body:
                    yield False, " ".join(body)
                    body = []
                if is_letter_spaced(text):
                    text = clean(" ".join(unspaced_line(line) for line in run))
                yield True, text
                continue
            body.append(text)
        else:
            body.append(clean(line_text(lines[index])))
            index += 1
    if body:
        yield False, " ".join(body)


# ------------------------------------------------------------------- chunking

# A numbered step: "1) Shuffle the deck". The lookbehind keeps cross-references
# like "(see page 9)" and parentheticals like "(1) facedown" from matching, and
# the capital letter after the bracket is what a real step always starts with.
STEP = re.compile(r"(?<![\d(])\d{1,2}\)\s+(?=[A-Z])")

# End of a sentence: closing punctuation, a space, then something that is not a
# lower-case letter. That last part keeps "e.g. the" and "approx. 3" together.
SENTENCE = re.compile(r"(?<=[.!?”])\s+(?=[^a-z])")


def split_steps(text):
    """Split a paragraph in front of each numbered step, keeping the number."""
    cuts = [match.start() for match in STEP.finditer(text)]
    pieces = (text[start:end] for start, end in zip([0] + cuts, cuts + [len(text)]))
    return [piece.strip() for piece in pieces if piece.strip()]


def sentences(paragraphs):
    """Yield (page, group, sentence) for everything in one section.

    `group` changes at every numbered step, and again at the first text that is
    not one. A chunk never spans two groups, which is what keeps each step
    standing on its own. Sentences longer than a whole chunk are wrapped on
    spaces, so no chunk ever ends mid-word.
    """
    group, after_step = 0, False
    for page, paragraph in paragraphs:
        for piece in split_steps(paragraph):
            is_step = STEP.match(piece) is not None
            if is_step or after_step:
                group += 1
            after_step = is_step
            for sentence in SENTENCE.split(piece):
                for part in textwrap.wrap(
                    sentence, CHUNK_MAX, break_long_words=False, break_on_hyphens=False
                ):
                    yield page, group, part


def chunk(paragraphs):
    """Pack one section into chunks of at most CHUNK_MAX characters.

    Sentences are filled in until the next one would not fit, so a chunk only
    ever ends at a sentence, paragraph, or step boundary.

    Yields (page, text), where page is the page the chunk starts on.
    """
    current, page, group = "", None, None
    for sentence_page, sentence_group, sentence in sentences(paragraphs):
        if current and (sentence_group != group or len(current) + 1 + len(sentence) > CHUNK_MAX):
            yield page, current
            current = ""
        if not current:
            page, group = sentence_page, sentence_group
        current = f"{current} {sentence}".strip()
    if current:
        yield page, current


# -------------------------------------------------------------------- ingest

def label(game, section, text):
    """Prefix a chunk so the embedding carries the game and section with it.

    A chunk that is nothing but its own title (a component list, a diagram
    label) would otherwise read as the same words twice.
    """
    if not section:
        return f"{game}: {text}"
    if section == text:
        return f"{game} › {section}"
    return f"{game} › {section}: {text}"


def embed(texts):
    vectors = []
    for start in range(0, len(texts), EMBED_BATCH):
        batch = texts[start:start + EMBED_BATCH]
        vectors.extend(ollama.embed(model=EMBED_MODEL, input=batch).embeddings)
    return vectors


def main():
    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"no PDFs in {PDF_DIR}")

    if DB_DIR.exists():
        shutil.rmtree(DB_DIR)  # ingest always rebuilds, so reruns cannot double up
    client = chromadb.PersistentClient(path=str(DB_DIR))
    collection = client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})

    for pdf in pdfs:
        # Files are named <game>_<source>.pdf, e.g. mars_faq.pdf.
        game, _, source = pdf.stem.rpartition("_")
        game = game.replace("_", " ").title()

        documents, metadatas, ids = [], [], []
        for section, paragraphs in read_sections(pdf, source):
            for page, text in chunk(paragraphs):
                if not any(c.isalpha() for c in text):
                    continue  # a bare number or symbol, e.g. a contents page number
                documents.append(label(game, section, text))
                metadatas.append({"game": game, "source": source, "page": page, "section": section})
                ids.append(f"{pdf.stem}-{len(ids)}")

        print(f"{pdf.name}: {len(documents)} chunks, embedding...")
        collection.add(ids=ids, documents=documents, embeddings=embed(documents), metadatas=metadatas)

    print(f"stored {collection.count()} chunks in {DB_DIR}")


if __name__ == "__main__":
    main()
