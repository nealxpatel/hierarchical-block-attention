"""Data pipeline: tokenizes a quality open web/books/code mix with the donor's own
tokenizer into uint32 binary shards (vocab sizes above 65536, as most current
tokenizers use, don't fit uint16). Run wherever there's network access to pull
FineWeb-Edu / code streams; --smoke builds tiny offline shards from a local
wikitext cache + the corpus directory, for plumbing-testing heal.py / evals.py
without network access.

Shards (under $HBA_DATA_DIR, default ./data/):
  train.bin (web+books+code mix, <eos>-delimited docs)
  val_web.bin / val_books.bin / val_code.bin (held-out PPL)
  needle_books.bin (held-out book haystack for the needle eval -- see below)
  meta.json

HOLDOUT DISCIPLINE: each source is ONE iterator; the val shard is encoded FIRST
from the head of the stream, then the train shard continues on the SAME iterator
-- val/train are disjoint by construction. The needle haystack must be a book
that is NEVER trained on (a real prep refuses to run without one; see
--needle-book below) -- silently falling back to training text would fake the
needle holdout that docs/evals.md's needle sweep depends on.

Full mix (default budgets, ~300-500M train tokens):
  web   FineWeb-Edu sample-10BT (streaming, parquet-native)                ~60%
  books corpus_dir/*.txt (the needle-holdout book excluded) + wikitext-103 ~20%
  code  codeparrot/codeparrot-clean (Python, json.gz, script-free) stream  ~20%
  NOTE: codeparrot/github-code-clean is a SCRIPT dataset and fails under
  datasets>=3 ("Dataset scripts are no longer supported") -- do not swap it
  back in.

Usage:
  python -m hba.data_prep --train-tokens 4e8
  python -m hba.data_prep --smoke                    # tiny, offline
"""

import argparse
import glob
import json
import os
import sys
import time

import numpy as np

from .config import CORPUS_DIR, DATA, DONOR_NAME

EOS = None   # set from tokenizer


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def get_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(DONOR_NAME)


def encode_stream(tok, texts, out_path, cap_tokens):
    """Encode from an iterable of documents to uint32 .bin, <eos>-delimited; stops
    soon after cap_tokens. The iterator is NOT reset -- a later call on the same
    iterator continues where this one stopped (disjoint shards). Returns token
    count."""
    global EOS
    if EOS is None:
        EOS = tok.eos_token_id if tok.eos_token_id is not None else 151643
    # small flush/write granularity so small caps (val shards) don't massively overshoot
    chunk = 1_000_000 if not cap_tokens else max(50_000, min(1_000_000, int(cap_tokens) // 4))
    buf, written = [], 0
    f = open(out_path, "wb")
    batch, batch_chars = [], 0

    def flush():
        nonlocal batch_chars
        if not batch:
            return
        for ids in tok(batch)["input_ids"]:
            buf.extend(ids); buf.append(EOS)
        batch.clear(); batch_chars = 0
    for t in texts:
        if not t or not t.strip():
            continue
        batch.append(t); batch_chars += len(t)
        # flush per 32 docs OR ~200K chars so a whole-book doc can't blow a small
        # val cap by orders of magnitude before the cap check fires
        if len(batch) >= 32 or batch_chars >= 200_000:
            flush()
        if len(buf) >= chunk:
            arr = np.asarray(buf, dtype=np.uint32)
            f.write(arr.tobytes()); written += arr.size; buf.clear()
            if cap_tokens and written >= cap_tokens:
                break
    flush()
    if buf:
        arr = np.asarray(buf, dtype=np.uint32)
        f.write(arr.tobytes()); written += arr.size
    f.close()
    return written


# ---------------------------------------------------------------- sources ------
def web_texts(smoke):
    from datasets import load_dataset
    if smoke:
        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
        for i in range(0, min(len(ds), 8000)):
            yield ds[i]["text"]
        return
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    for r in ds:
        yield r["text"]


def book_texts(smoke, corpus_dir, needle_book):
    # corpus documents FIRST (so val_books, taken from the stream head, is real
    # book prose and the corpus is not silently starved out of training by the
    # wikitext token cap), then wikitext.
    for p in sorted(glob.glob(os.path.join(corpus_dir, "*.txt"))):
        if needle_book and needle_book in os.path.basename(p).lower():
            continue                            # held out for needles
        yield open(p, encoding="utf-8", errors="ignore").read()
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    take = 4000 if smoke else len(ds)
    for i in range(take):
        yield ds[i]["text"]


def code_texts(smoke):
    if smoke:
        import site
        roots = list(getattr(site, "getsitepackages", lambda: [])()) + [os.path.dirname(os.__file__)]
        n = 0
        for r in roots:
            for p in sorted(glob.glob(os.path.join(r, "**", "*.py"), recursive=True)):
                try:
                    t = open(p, encoding="utf-8", errors="ignore").read()
                except Exception:
                    continue
                if len(t) > 200:
                    yield t; n += 1
                if n > 400:
                    return
        return
    from datasets import load_dataset
    ds = load_dataset("codeparrot/codeparrot-clean", split="train", streaming=True)
    for r in ds:
        yield r["content"]


# ---------------------------------------------------------------- main ---------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-tokens", type=float, default=4e8)
    ap.add_argument("--web-frac", type=float, default=0.6)
    ap.add_argument("--book-frac", type=float, default=0.2)   # code = 1 - web - book
    ap.add_argument("--corpus-dir", default=CORPUS_DIR,
                    help="directory of *.txt documents for the books domain (default: "
                         "$HBA_CORPUS_DIR or <data>/corpus)")
    ap.add_argument("--needle-book", default="middlemarch",
                    help="substring (case-insensitive) matching the corpus .txt file to hold "
                         "out as the needle-eval haystack; required for a non-smoke run")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    meta_p = os.path.join(DATA, "meta.json")
    if args.smoke and os.path.exists(meta_p) and not json.load(open(meta_p)).get("smoke", True):
        print("REFUSING: full shards exist; --smoke would orphan trained checkpoints. rm data/ first.")
        sys.exit(2)

    needle_matches = [p for p in glob.glob(os.path.join(args.corpus_dir, "*.txt"))
                      if args.needle_book in os.path.basename(p).lower()]
    if not args.smoke and not needle_matches:
        print(f"ABORT: no *.txt in {args.corpus_dir} matching --needle-book "
              f"'{args.needle_book}' -- the needle haystack MUST be a held-out book (put one "
              "in the corpus dir; a fallback to training text would fake the needle holdout).")
        sys.exit(2)
    needle_path = needle_matches[0] if needle_matches else None

    tok = get_tokenizer()
    T = 6e6 if args.smoke else args.train_tokens
    web_cap = int(args.web_frac * T)
    book_cap = int(args.book_frac * T)
    code_cap = int((1 - args.web_frac - args.book_frac) * T)
    log(f"train target {T/1e6:.0f}M tok: web {web_cap/1e6:.0f}M books {book_cap/1e6:.0f}M code {code_cap/1e6:.0f}M")

    web_it = web_texts(args.smoke)
    book_it = book_texts(args.smoke, args.corpus_dir, args.needle_book)
    code_it = code_texts(args.smoke)

    # ---- held-out val shards FIRST (stream heads), train continues on the same iterators ----
    log("encoding held-out val shards (stream heads; disjoint from train by construction)...")
    nvw = encode_stream(tok, web_it, os.path.join(DATA, "val_web.bin"),
                        200_000 if args.smoke else 1_000_000)
    nvb = encode_stream(tok, book_it, os.path.join(DATA, "val_books.bin"),
                        200_000 if args.smoke else 1_000_000)
    nvc = encode_stream(tok, code_it, os.path.join(DATA, "val_code.bin"),
                        100_000 if args.smoke else 500_000)
    log(f"  val: web {nvw/1e3:.0f}K books {nvb/1e3:.0f}K code {nvc/1e3:.0f}K")

    log("encoding web..."); nw = encode_stream(tok, web_it, os.path.join(DATA, "_web.bin"), web_cap)
    log(f"  web {nw/1e6:.1f}M tok")
    log("encoding books..."); nb = encode_stream(tok, book_it, os.path.join(DATA, "_books.bin"), book_cap)
    log(f"  books {nb/1e6:.1f}M tok")
    log("encoding code..."); nc = encode_stream(tok, code_it, os.path.join(DATA, "_code.bin"), code_cap)
    log(f"  code {nc/1e6:.1f}M tok")

    with open(os.path.join(DATA, "train.bin"), "wb") as out:
        for part in ("_web.bin", "_books.bin", "_code.bin"):
            pp = os.path.join(DATA, part)
            with open(pp, "rb") as f:
                out.write(f.read())
            os.remove(pp)
    n_train = nw + nb + nc

    # needle haystack: the held-out book ONLY (never trained on; hard-required above)
    if needle_path:
        nn = encode_stream(tok, [open(needle_path, encoding="utf-8", errors="ignore").read()],
                           os.path.join(DATA, "needle_books.bin"), None)
    else:  # smoke-only path
        log("SMOKE: no needle book found -- needle shard falls back to the held-out book "
            "stream (plumbing only; a REAL prep refuses this above)")
        nn = encode_stream(tok, book_it, os.path.join(DATA, "needle_books.bin"), 600_000)

    meta = dict(vocab_size=len(tok), eos=EOS, dtype="uint32", n_train_tokens=int(n_train),
                n_web=int(nw), n_books=int(nb), n_code=int(nc),
                n_val_web=int(nvw), n_val_books=int(nvb), n_val_code=int(nvc), n_needle=int(nn),
                holdout="stream-head val shards, disjoint from train; needle=held-out book",
                smoke=args.smoke)
    json.dump(meta, open(meta_p, "w"), indent=2)
    log(f"DONE train={n_train/1e6:.1f}M tok (web {nw/1e6:.0f} books {nb/1e6:.0f} code {nc/1e6:.0f}) "
        f"needle={nn/1e3:.0f}K -> {meta_p}")


if __name__ == "__main__":
    main()
