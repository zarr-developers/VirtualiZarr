#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Build an Euler diagram of format cloud-suitability from a membership table.

The table (CSV or a markdown table) is a boolean membership matrix: rows are
formats, columns are categories. This derives the containment poset, verifies
that the categories form a total order (a strict chain) plus one disjoint set,
and renders the chain as nested rounded rectangles with the disjoint set as its
own panel -- one label block per non-empty region, and no region drawn that has
no members. If the table stops having that shape, it refuses to draw rather
than emit a diagram that lies about the structure.

Rectangles rather than ellipses: a rectangle offers its full width to text at
every height, so labels lay out on a grid instead of being squeezed into the
narrowing top of an oval.

The output is self-contained and carries both a dark look (the default, and
what a standalone view gets) and a light one that activates under
mkdocs-material's default colour scheme, so an inlined copy follows the host
page's own light/dark toggle rather than the reader's OS setting.

Usage:
    ./format_tiers.py [TABLE] [-o OUT.svg]

Text is set in the host page's own typeface; only a generic fallback stack is
declared here.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

HERE = Path(__file__).parent

TRUTHY = {"x", "X", "1", "true", "TRUE", "yes", "Y", "y", "✓"}

# The set that is disjoint from the nested chain. It gets its own panel, set
# apart -- never enclosing the chain, which would assert a containment the
# table denies.
DISJOINT = "Non-cloud-optimizable"

# One hue per category, listed in nesting order -- which is also the order
# adjacent panels appear in, the pairs that have to stay distinguishable.
# Checked against a dark surface for chroma, for contrast (all >= 3:1), and for
# separation under protanopia, deuteranopia and tritanopia. The accents are
# deliberately bright: they are chosen to sit on the dark surface below.
COLORS = {
    "Non-cloud-optimizable": "#FF6554",  # red
    "Cloud-optimizable via virtualization": "#A653FF",  # violet
    "Cloud-optimizable upon write": "#FF9E0D",  # orange
    "Cloud-optimized by default": "#F881D1",  # pink
    "Cloud-Native (static)": "#31D495",  # green
    "Cloud-Native (transactional)": "#5EC4F7",  # blue
}

DARK_SURFACE = "#201F2C"
LIGHT_SURFACE = "#F5F5F5"

# No face is shipped or named: the figure inherits whatever the host page sets,
# so it matches the surrounding text rather than importing a second typeface.
FONT_STACK = "inherit, ui-sans-serif, sans-serif"

ITEM_W, NAME_W = 400, 500  # which weight each text role uses

NAME_FS = 15  # set-name label
ITEM_FS = 13  # format label
LINE_H = 22
NAME_GAP = 11  # set name baseline -> first item baseline
PAD_TOP = 26  # a panel's top edge -> its name baseline
PAD_BOT = 16  # last item baseline -> the next panel's top edge
MAX_COLS = 4

INSET = 17  # how far each nested panel sits inside its parent
CHAIN_W = 720.0
APART_W = 400.0
GROUP_GAP = 64  # clear space between the two disjoint groups
MARGIN = 40

TITLE = "Cloud suitability of various formats"
TITLE_FS = 22
TITLE_TRACK = -0.03  # -3% tracking on the title
HEAD = 40  # space the title occupies above the panels


# ---------------------------------------------------------------- parse


def parse_table(path: Path) -> tuple[list[str], dict[str, set[str]]]:
    """Read a membership matrix from CSV or a markdown table."""
    if path.suffix.lower() == ".csv":
        with path.open(newline="") as f:
            rows = [r for r in csv.reader(f) if r and any(c.strip() for c in r)]
    else:
        lines = [
            ln.strip()
            for ln in path.read_text().splitlines()
            if ln.strip().startswith("|")
        ]
        rows = [[c.strip() for c in ln.strip().strip("|").split("|")] for ln in lines]
        rows = [r for r in rows if set("".join(r)) - set("-: ")]  # drop the ---- rule
    if len(rows) < 2:
        sys.exit(f"{path}: no table found")

    categories = [c.strip() for c in rows[0][1:]]
    members: dict[str, set[str]] = {c: set() for c in categories}
    items: list[str] = []
    for row in rows[1:]:
        item = row[0].strip()
        if not item:
            continue
        items.append(item)
        for cat, val in zip(categories, row[1:]):
            if val.strip() in TRUTHY:
                members[cat].add(item)
    return items, members


# ------------------------------------------------------- structure check


def derive_chain(
    items: list[str], membership: dict[str, set[str]]
) -> tuple[list[str], list[str]]:
    """Return (chain outermost->innermost, uncategorised items)."""
    if DISJOINT not in membership:
        sys.exit(f"expected a column named {DISJOINT!r}")

    apart = membership[DISJOINT]
    inner = {k: v for k, v in membership.items() if k != DISJOINT}

    for name, s in inner.items():
        if apart & s:
            shared = ", ".join(sorted(apart & s))
            sys.exit(
                f"{DISJOINT!r} overlaps {name!r} on: {shared}\n"
                "That set is drawn apart, so it must share no members."
            )

    # A chain sorts by size; verify each is a strict subset of its successor.
    chain = sorted(inner, key=lambda k: (-len(inner[k]), k))
    for outer, nxt in zip(chain, chain[1:]):
        if not inner[nxt] <= inner[outer]:
            stray = ", ".join(sorted(inner[nxt] - inner[outer]))
            sys.exit(
                f"not a chain: {nxt!r} is not a subset of {outer!r}\n"
                f"  in {nxt} but not {outer}: {stray}\n"
                "Partial overlaps need an Euler solver (R's eulerr), not "
                "nested panels."
            )

    missing = [c for c in [DISJOINT] + chain if c not in COLORS]
    if missing:
        sys.exit(f"no colour assigned for: {', '.join(missing)}")

    covered = apart | (inner[chain[0]] if chain else set())
    orphans = [i for i in items if i not in covered]
    return chain, orphans


# ------------------------------------------------------------- metrics
#
# Advance widths are estimated from a per-weight ratio, since the rendering
# typeface is whatever the host page uses and is not known here.


RATIO = {400: 0.545, 500: 0.56}



def text_w(s: str, fs: float, weight: int = ITEM_W) -> float:
    """Estimated advance width of s at the given size and weight."""
    return len(s) * fs * RATIO[weight]



# ------------------------------------------------------------- geometry


def fit_cols(labels: list[str], avail: float) -> int:
    """Widest column count whose longest label still fits."""
    widest = max((text_w(s, ITEM_FS) for s in labels), default=0)
    for cols in range(min(MAX_COLS, len(labels)), 1, -1):
        if widest <= (avail / cols) * 0.86:
            return cols
    return 1


def grid(labels: list[str], avail: float) -> tuple[int, int]:
    cols = fit_cols(labels, avail)
    return cols, math.ceil(len(labels) / cols)


def block_h(labels: list[str], avail: float) -> float:
    """Height a label block needs: top pad, name, item rows, bottom pad."""
    _, rows = grid(labels, avail)
    return PAD_TOP + NAME_GAP + rows * LINE_H + PAD_BOT


# ---------------------------------------------------------------- render


def mix(hex_color: str, other: str, t: float) -> str:
    """Blend hex_color toward `other` by t (0..1)."""
    a = [int(hex_color[i : i + 2], 16) for i in (1, 3, 5)]
    b = [int(other[i : i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(x + (y - x) * t):02X}" for x, y in zip(a, b))


def esc(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def slug(name: str) -> str:
    return "".join(ch if ch.isalnum() else "-" for ch in name.lower()).strip("-")


def theme_rules(names: list[str], prefix: str, dark: bool) -> list[str]:
    """Surface, ink and per-set colours for one colour scheme."""
    surface, ink = (DARK_SURFACE, LIGHT_SURFACE) if dark else (LIGHT_SURFACE, DARK_SURFACE)
    rules = [
        # Adopt the host page's background when inlined, so the figure has no
        # visible block of its own; falls back to the scheme surface
        # standalone, which is what the PNG export gets.
        f"{prefix}.eu-surface {{ fill: var(--md-default-bg-color, {surface}); }}",
        f"{prefix}.eu-ink {{ fill: {ink}; }}",
    ]
    for name in names:
        c, k = COLORS[name], slug(name)
        if dark:
            # Accents at full strength, with the set name lifted toward white
            # so it reads as a heading against its own panel.
            edge, label = c, mix(c, "#FFFFFF", 0.2)
        else:
            # The same hues darkened: the accents above are chosen for the
            # dark surface and would not hold contrast on the light one.
            edge, label = mix(c, DARK_SURFACE, 0.28), mix(c, DARK_SURFACE, 0.45)
        rules += [
            f"{prefix}.eu-fill-{k} {{ fill: {mix(c, surface, 0.86)}; }}",
            f"{prefix}.eu-edge-{k} {{ stroke: {edge}; }}",
            f"{prefix}.eu-name-{k} {{ fill: {label}; }}",
        ]
    return rules


def style_block(names: list[str]) -> str:
    """Dark by default; light only under mkdocs-material's default scheme.

    Dark is the base so a standalone view and the PNG export are dark without
    depending on the viewer's OS. The attribute selectors carry more
    specificity than the bare classes, so when this SVG is inlined into
    mkdocs-material the page's own toggle always wins over the OS setting.
    """
    css = []
    css.append(f"  .eu-text {{ font-family: {FONT_STACK}; }}")
    css.append("  [class^='eu-edge-'] { fill: none; stroke-width: 2; }")
    css += ["  " + r for r in theme_rules(names, "", dark=True)]
    css += [
        "  " + r
        for r in theme_rules(names, '[data-md-color-scheme="default"] ', dark=False)
    ]
    return "\n".join(css)


def panel(parts: list[str], name: str, x: float, y: float, w: float, h: float, r: float):
    k = slug(name)
    for cls in (f"eu-fill-{k}", f"eu-edge-{k}"):
        parts.append(
            f'<rect class="{cls}" x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" '
            f'height="{h:.1f}" rx="{r:.1f}"/>'
        )


def label_block(
    parts: list[str], name: str, labels: list[str], cx: float, w: float, y_name: float
) -> None:
    """Emit a set's name and its member list on a grid centred at cx."""
    parts.append(
        f'<text class="eu-text eu-name-{slug(name)}" x="{cx:.1f}" y="{y_name:.1f}" '
        f'text-anchor="middle" font-size="{NAME_FS}" font-weight="{NAME_W}">'
        f"{esc(name)}</text>"
    )
    cols, _ = grid(labels, w)
    colw = w * 0.9 / cols
    y0 = y_name + NAME_GAP + ITEM_FS
    for i, label in enumerate(labels):
        row, col = divmod(i, cols)
        # Centre each row on its own, so a short final row sits under the
        # middle of the block instead of hanging off to the left.
        in_row = min(cols, len(labels) - row * cols)
        tx = cx + (col - (in_row - 1) / 2) * colw
        parts.append(
            f'<text class="eu-text eu-ink" x="{tx:.1f}" '
            f'y="{y0 + row * LINE_H:.1f}" text-anchor="middle" '
            f'font-size="{ITEM_FS}" font-weight="{ITEM_W}">{esc(label)}</text>'
        )


def build_svg(
    chain: list[str], membership: dict[str, set[str]], items: list[str]
) -> str:
    order = {it: n for n, it in enumerate(items)}
    key = lambda s: sorted(s, key=lambda it: order.get(it, 1 << 30))  # noqa: E731

    # Each ring's band is what it does not share with the ring inside it.
    bands = [
        key(membership[nm] - (membership[chain[i + 1]] if i + 1 < len(chain) else set()))
        for i, nm in enumerate(chain)
    ]
    n = len(chain)

    # Nested panels: each is INSET narrower on both sides than its parent, and
    # starts below its parent's label block.
    widths = [CHAIN_W - 2 * INSET * i for i in range(n)]
    heights = [block_h(bands[i], widths[i]) for i in range(n)]

    top = float(MARGIN + HEAD)
    tops = [top]
    for h in heights[:-1]:
        tops.append(tops[-1] + h)

    chain_bottom = tops[-1] + heights[-1] + INSET * (n - 1)
    chain_h = chain_bottom - top
    fig_h = chain_bottom + MARGIN
    fig_w = MARGIN * 2 + APART_W + GROUP_GAP + CHAIN_W

    apart_x = float(MARGIN)
    chain_x = MARGIN + APART_W + GROUP_GAP

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {fig_w:.0f} '
        f'{fig_h:.0f}" width="{fig_w:.0f}" height="{fig_h:.0f}" role="img" '
        f'aria-label="Euler diagram: cloud suitability of array and file '
        f'formats. {esc(" ; ".join(f"{c} contains {len(membership[c])} formats" for c in [DISJOINT] + chain))}">',
        "<style>\n" + style_block([DISJOINT] + chain) + "\n</style>",
        f'<rect class="eu-surface" x="0" y="0" width="{fig_w:.0f}" '
        f'height="{fig_h:.0f}"/>',
    ]

    # The disjoint set, in its own panel -- it encloses nothing and nothing
    # encloses it. Sized to match the chain's height so neither group reads as
    # subordinate to the other.
    apart_items = key(membership[DISJOINT])
    panel(parts, DISJOINT, apart_x, top, APART_W, chain_h, 20)
    label_block(
        parts,
        DISJOINT,
        apart_items,
        apart_x + APART_W / 2,
        APART_W,
        top + (chain_h - block_h(apart_items, APART_W)) / 2 + PAD_TOP,
    )

    # The chain, outermost first so each nested panel paints over its parent.
    for i, name in enumerate(chain):
        panel(
            parts,
            name,
            chain_x + INSET * i,
            tops[i],
            widths[i],
            chain_bottom - INSET * i - tops[i],
            max(9.0, 20 - 2.5 * i),
        )
    for i, name in enumerate(chain):
        label_block(
            parts, name, bands[i], chain_x + CHAIN_W / 2, widths[i], tops[i] + PAD_TOP
        )

    parts.append(
        f'<text class="eu-text eu-ink" x="{fig_w / 2:.1f}" y="{MARGIN + 6}" '
        f'text-anchor="middle" font-size="{TITLE_FS}" font-weight="{NAME_W}" '
        f'letter-spacing="{TITLE_FS * TITLE_TRACK:.2f}">{esc(TITLE)}</text>'
    )
    parts.append("</svg>")
    return "\n".join(parts)


def render(table: Path | str | None = None) -> str:
    """Return the SVG markup. The entry point for a docs build."""
    path = Path(table) if table else HERE / "format_tiers.md"
    items, membership = parse_table(path)
    chain, _ = derive_chain(items, membership)
    return build_svg(chain, membership, items)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("table", nargs="?", default=HERE / "format_tiers.md", type=Path)
    ap.add_argument("-o", "--out", default=HERE / "euler.svg", type=Path)
    args = ap.parse_args()

    items, membership = parse_table(args.table)
    chain, orphans = derive_chain(items, membership)

    print(f"{len(items)} formats, {len(chain) + 1} categories")
    print(f"  {len(membership[DISJOINT]):>3}  {DISJOINT}  (disjoint, drawn apart)")
    for name in chain:
        print(f"  {len(membership[name]):>3}  {name}")
    if orphans:
        print(f"\nWARNING: uncategorised rows (drawn nowhere): {', '.join(orphans)}")
    print("\nverified: strict total order, no partial overlaps")

    args.out.write_text(build_svg(chain, membership, items))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
