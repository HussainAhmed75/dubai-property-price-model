"""Checks that every number written in the given text files exists in
results/metrics.json (at the precision it is written).

Usage:
    python scripts/verify_numbers.py README.md [more files ...]

Ignored on purpose: years (1990-2030), ISO dates, numbers inside URLs, code
spans/blocks, file names, and list markers ("1." at line start).
Exit code 1 if any number cannot be traced back to metrics.json.
"""
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def leaves(o):
    if isinstance(o, dict):
        for v in o.values():
            yield from leaves(v)
    elif isinstance(o, list):
        for v in o:
            yield from leaves(v)
    elif isinstance(o, bool):
        return
    elif isinstance(o, (int, float)):
        yield float(o)
    elif isinstance(o, str):
        # numbers embedded in strings (e.g. dates are ignored later anyway)
        for m in re.findall(r"-?\d+(?:\.\d+)?", o):
            yield float(m)


NUM = re.compile(r"(?<![\w.])([+\-−]?)(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?\s*([kKM]\b)?")


def strip(text):
    text = re.sub(r"```.*?```", " ", text, flags=re.S)
    text = re.sub(r"`[^`]*`", " ", text)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"\]\([^)]*\)", "]", text)                       # markdown link targets
    text = re.sub(r"<[^>]+>", " ", text)                             # html tags
    text = re.sub(r"\b[\w\-./]+\.(png|pdf|csv|json|py|ipynb|html|txt|md|gz|parquet)\b", " ", text)
    text = re.sub(r"\b\d{4}-\d{2}-\d{2}\b", " ", text)               # ISO dates
    text = re.sub(r"\b\d{4}-\d{2}\b", " ", text)                     # year-month
    text = re.sub(r"(?m)^\s*\d+[.)]\s", " ", text)                   # list markers
    text = re.sub(r"#\d+", " ", text)
    return text


def check(path, values):
    raw = open(path, encoding="utf-8").read()
    text = strip(raw)
    bad, ok = [], 0
    for m in NUM.finditer(text):
        sign, ip, dp, suf = m.groups()
        s = ip.replace(",", "") + (dp or "")
        t = float(s)
        dec = len(dp) - 1 if dp else 0
        if not dp and not suf and 1990 <= t <= 2030:
            continue                                   # a year
        scale = {"k": 1e3, "K": 1e3, "M": 1e6}.get(suf or "", 1)
        found = False
        for v in values:
            for cand in (v, abs(v)):
                if scale == 1:
                    if abs(round(cand, dec) - t) < 1e-9:
                        found = True
                else:
                    if abs(round(cand / scale, dec) - t) < 1e-9:
                        found = True
            if found:
                break
        if found:
            ok += 1
        else:
            ctx = text[max(0, m.start() - 40): m.end() + 30].replace("\n", " ")
            bad.append((m.group(0).strip(), ctx))
    return ok, bad


def main(files):
    metrics = json.load(open(os.path.join(ROOT, "results", "metrics.json"), encoding="utf-8"))
    values = list(set(leaves(metrics)))
    failed = False
    for f in files:
        ok, bad = check(f, values)
        print(f"{f}: {ok} numbers matched metrics.json, {len(bad)} unmatched")
        for b in bad:
            failed = True
            print(f"   UNMATCHED {b[0]!r}  … {b[1]} …")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main(sys.argv[1:] or [os.path.join(ROOT, "README.md")])
