"""Map each figure and table in the compiled manuscript to what produces it.

Reviewers read the PDF, where assets are numbered, while this repository names
them by content. This joins the two, so "Fig 15" can be traced to the target that
makes it without renaming a single file.

The manuscript is the source of truth for the numbering. Anything inside
\\cut{...} is swallowed by \\newcommand{\\cut}[1]{} and so is not in the paper and
takes no number; commented lines are likewise skipped. Numbering counts both
`figure` and `wrapfigure` environments, in order.

Which target writes which file is read from a run log of generate_figures.py,
because the export scripts print every path they write.

Usage (from the repository root):

    python3 scripts/build_figure_index.py \\
        --tex /path/to/plos_latex_template.tex \\
        --log logs/make_figures_<jobid>.out

Writes FIGURES.md.
"""
import argparse
import json
import os
import re
import sys

ROOT = os.environ.get("POMDP_ROOT") or os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def live_text(path):
    """The manuscript with commented lines and every \\cut{...} region removed."""
    lines = open(path, errors="ignore").read().split("\n")
    for i, line in enumerate(lines):
        lines[i] = "" if line.lstrip().startswith("%") else re.sub(r"(?<!\\)%.*$", "", line)
    text = "\n".join(lines)
    out = list(text)
    for match in re.finditer(r"\\cut\{", text):
        depth = 0
        for j in range(match.end() - 1, len(text)):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    for k in range(match.start(), j + 1):
                        out[k] = " "
                    break
    return "".join(out)


def letters(n):
    """1 -> A, 26 -> Z, 27 -> AA, matching how the appendix labels its floats."""
    out = ""
    while n > 0:
        n, r = divmod(n - 1, 26)
        out = chr(ord("A") + r) + out
    return out


def numbered_assets(text, appendix_marker):
    """(kind, number, label, [files], in_appendix) for every surviving float.

    The appendix counts separately and in letters, so the paper's Fig 1 and the
    appendix's Fig A are different assets and neither shifts when the other
    gains a figure.
    """
    assets = []
    counts = {("figure", False): 0, ("figure", True): 0,
              ("table", False): 0, ("table", True): 0}
    current, in_appendix = None, False
    for line in text.split("\n"):
        if appendix_marker and appendix_marker in line:
            in_appendix = True
        begin = re.match(r"\s*\\begin\{(figure|wrapfigure|table)\*?\}", line)
        if begin:
            kind = "table" if begin.group(1) == "table" else "figure"
            counts[(kind, in_appendix)] += 1
            n = counts[(kind, in_appendix)]
            current = [kind, letters(n) if in_appendix else str(n), None, [],
                       in_appendix]
            assets.append(current)
        if current is None:
            continue
        for f in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", line):
            current[3].append(f)
        for f in re.findall(r"\\input\{([^}]+)\}", line):
            current[3].append(f)
        label = re.search(r"\\label\{([^}]+)\}", line)
        if label and current[2] is None:
            current[2] = label.group(1)
    return assets


def target_outputs(log_path, stems):
    """stem -> target, by finding each known asset name inside a target's block.

    The export scripts announce their output in no common way: some print a
    path, some a directory, some just the name and the figure size. Rather than
    teach the parser every phrasing, split the log into per-target blocks and
    look for the names we already know the manuscript uses.
    """
    produced, target, block = {}, None, []

    def flush():
        if not target:
            return
        text = "\n".join(block)
        for stem in stems:
            if stem and stem in text:
                produced.setdefault(stem, target)

    for line in open(log_path, errors="ignore"):
        header = re.match(r"=== (\S+)\s+\(", line)
        if header:
            flush()
            target, block = header.group(1), []
        else:
            block.append(line.rstrip())
    flush()
    return produced


def notebook_outputs(stems, root):
    """stem -> notebook, for the assets the comparison notebooks write.

    Several tables and figures come from notebooks rather than from a figure
    target, which PIPELINE.txt section 5.2 describes. Saying which notebook is
    more use to a reader than saying nothing.
    """
    import glob
    found = {}
    for path in sorted(glob.glob(os.path.join(root, "notebooks", "*.ipynb"))):
        try:
            nb = json.load(open(path, errors="ignore"))
        except Exception:
            continue
        text = "\n".join("".join(c.get("source", ""))
                          for c in nb.get("cells", []) if c.get("cell_type") == "code")
        for stem in stems:
            if stem and stem in text:
                found.setdefault(stem, "notebooks/" + os.path.basename(path))
    return found


def resolve(ref, stem):
    """The file the manuscript means, found by its path rather than its name.

    Many stems exist in several trees at once: long_models_metrics.tex is in
    BIC/tables, BIC_commit/tables and BIC/tables_corrected_bic. Searching by
    filename would pick whichever the walk reached first, so the manuscript's
    own relative path decides, and a name search is only the last resort.
    """
    rel = ref[len("figures/"):] if ref.startswith("figures/") else ref
    for base in (os.path.join(ROOT, rel), os.path.join(ROOT, "figures", rel),
                 os.path.join(ROOT, ref)):
        for candidate in (base, base + ".pdf", base + ".png", base + ".tex"):
            if os.path.isfile(candidate):
                return candidate
    for root, dirs, files in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in (".git", "__pycache__")]
        for f in files:
            if f.split(".")[0] == stem and f.endswith((".pdf", ".png", ".tex")):
                return os.path.join(root, f)
    return None


def collect(rows, dest):
    """Copy each asset out under its paper number, keeping its own name.

    A reader holding only the PDF needs Fig 15; whoever maintains the LaTeX needs
    optimality_gap. The copy carries both, and the originals are untouched.
    """
    import shutil
    os.makedirs(dest, exist_ok=True)
    found = missing = 0
    for kind, number, stem, _target, in_appendix, ref in rows:
        prefix = "Fig" if kind == "figure" else "Table"
        # zero pad the numeric ones so a directory listing reads in paper order
        tag = f"{int(number):02d}" if number.isdigit() else number
        source = resolve(ref, stem)
        if not source:
            missing += 1
            continue
        ext = os.path.splitext(source)[1]
        shutil.copy2(source, os.path.join(dest, f"{prefix}{tag}_{stem}{ext}"))
        found += 1
    print(f"  collected {found} assets into {os.path.relpath(dest, ROOT)}"
          + (f", {missing} with no file here" if missing else ""))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tex", required=True, help="the manuscript .tex")
    ap.add_argument("--log", help="a run log of generate_figures.py")
    ap.add_argument("--appendix", default=r"\section*{Supporting information}",
                    help="the line that starts the appendix, after which floats "
                         "count separately and in letters")
    ap.add_argument("--out", default=os.path.join(ROOT, "FIGURES.md"))
    ap.add_argument("--collect", metavar="DIR", nargs="?",
                    const=os.path.join(ROOT, "figures", "paper"),
                    help="also copy each asset out under its paper number, as "
                         "Fig01_<name>.pdf, so a reader holding only the PDF can "
                         "find it. The originals keep their own names.")
    args = ap.parse_args()

    assets = numbered_assets(live_text(args.tex), args.appendix)
    stems = {os.path.basename(f).split(".")[0]
             for _k, _n, _l, files, _a in assets for f in files}
    produced = target_outputs(args.log, stems) if args.log else {}
    from_nb = notebook_outputs(stems, ROOT)
    for stem, nb in from_nb.items():
        produced.setdefault(stem, nb)

    rows = []
    for kind, number, label, files, in_appendix in assets:
        for f in files:
            stem = os.path.basename(f).split(".")[0]
            rows.append((kind, number, stem, produced.get(stem, ""),
                         in_appendix, f))

    unknown = sum(1 for r in rows if not r[3])
    with open(args.out, "w") as fh:
        fh.write("# What produces each figure and table in the paper\n\n"
                 "The manuscript numbers its assets; this repository names them by\n"
                 "content. Three columns: the number as it appears in the paper, the\n"
                 "file name on disk, and what produces it.\n\n"
                 "Pass the **third** column to `--only`, not the file name:\n\n"
                 "    python3 scripts/generate_figures.py --only task_figure\n\n"
                 "Rows naming a notebook are produced by opening that notebook, not by\n"
                 "the figure runner. Rows saying `see PIPELINE.txt` come from a stage\n"
                 "documented there rather than from a single target.\n\n"
                 "The appendix counts separately and in letters, as the paper does, so\n"
                 "Fig 1 and Fig A are different assets. Generated by\n"
                 "`scripts/build_figure_index.py` from the manuscript, so the numbering\n"
                 "follows whatever the paper currently compiles.\n\n")
        for in_appendix, where in ((False, "Main text"), (True, "Appendix")):
            for kind, title in (("figure", "Figures"), ("table", "Tables")):
                sel = [r for r in rows if r[0] == kind and r[4] == in_appendix]
                if not sel:
                    continue
                fh.write(f"## {where}: {title.lower()}\n\n"
                         "| in the paper | file on disk | produced by |\n|---|---|---|\n")
                for _k, number, stem, target, _a, _ref in sel:
                    name = "Fig" if kind == "figure" else "Table"
                    fh.write(f"| {name} {number} | `{stem}` | "
                             f"{'`' + target + '`' if target else 'see PIPELINE.txt'} |\n")
                fh.write("\n")

    if args.collect:
        collect(rows, args.collect)

    print(f"{len(rows)} assets, {len(rows) - unknown} traced to a target -> "
          f"{os.path.relpath(args.out, ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
