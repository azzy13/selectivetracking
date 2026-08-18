#!/usr/bin/env python3
"""
Graphviz rendering of scene graphs from ``SceneGraphBuilder``.

``visualize_scene_graph.py`` draws the graph *on the image* — good for checking
that a node sits on the right pixels, bad for reading the relation structure,
because the edges all pile up in the middle of the frame.  This renders the same
frame graph as an actual graph: nodes with their attributes, edges labelled with
the relations that held.  That is the structure ``score_candidates`` walks, so
this is what you look at when a candidate scores the way it does.

    from scene_graph_dot import frame_graph_to_dot, render_dot, show_frame_graph

    show_frame_graph(frame_graph)                    # inline SVG in Jupyter
    render_dot(frame_graph_to_dot(fg), 'out.svg')    # to a file

CLI (renders every frame of a builder JSONL):

    python eval/scene_graph_dot.py --jsonl runs/sg.jsonl --out /tmp/viz
    python eval/scene_graph_dot.py --jsonl runs/sg.jsonl --out /tmp/viz --frames 0 5 --format png

Needs the ``dot`` binary on PATH (``graphviz`` package); the Python ``graphviz``
module is deliberately not a dependency — the DOT source is generated here and
piped straight to ``dot``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

from query_grounding import ROLE_ANCHOR, ROLE_TARGET


# ── palette ──────────────────────────────────────────────────────────────────
# Role drives the fill: an anchor is scaffolding, a target candidate is a
# possible answer, and that distinction is the whole point of the picture.
_ROLE_STYLE = {
    ROLE_TARGET: {"fill": "#e8f4ff", "border": "#2f6fb0", "head": "#2f6fb0"},
    ROLE_ANCHOR: {"fill": "#fff1de", "border": "#c8791a", "head": "#c8791a"},
}
_UNKNOWN_STYLE = {"fill": "#f0f0f0", "border": "#777777", "head": "#777777"}

# Attributes shown in the node body, in order.  Keys missing from the node are
# skipped rather than printed as "None".
_NODE_ATTRS = ("color", "size", "region", "motion", "confidence")


def _esc(s: Any) -> str:
    """Escape for a Graphviz HTML-like label."""
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def _fmt(value: Any) -> str:
    return f"{value:.3f}" if isinstance(value, float) else str(value)


def _node_label(
    node: Dict[str, Any],
    extra: Optional[Mapping[str, Any]] = None,
    attrs: Sequence[str] = _NODE_ATTRS,
) -> str:
    """HTML-like label: a coloured header with the id/role, then attribute rows."""
    style = _ROLE_STYLE.get(node.get("role", ROLE_TARGET), _UNKNOWN_STYLE)
    role = node.get("role", ROLE_TARGET)
    head = f"T{node['track_id']}  ({role.replace('_', ' ')})"

    rows = [
        f'<TR><TD COLSPAN="2" BGCOLOR="{style["head"]}">'
        f'<FONT COLOR="white"><B>{_esc(head)}</B></FONT></TD></TR>'
    ]
    for key in attrs:
        if node.get(key) is None:
            continue
        rows.append(
            f'<TR><TD ALIGN="LEFT"><FONT POINT-SIZE="9">{_esc(key)}</FONT></TD>'
            f'<TD ALIGN="LEFT"><B>{_esc(_fmt(node[key]))}</B></TD></TR>'
        )
    for key, value in (extra or {}).items():
        rows.append(
            f'<TR><TD ALIGN="LEFT"><FONT POINT-SIZE="9" COLOR="#777777">{_esc(key)}</FONT></TD>'
            f'<TD ALIGN="LEFT"><FONT COLOR="#777777">{_esc(_fmt(value))}</FONT></TD></TR>'
        )
    body = "".join(rows)
    return f'<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="1" CELLPADDING="2">{body}</TABLE>>'


def frame_graph_to_dot(
    frame_graph: Dict[str, Any],
    *,
    title: Optional[str] = None,
    scores: Optional[Mapping[int, float]] = None,
    node_notes: Optional[Mapping[int, Mapping[str, Any]]] = None,
    rankdir: str = "LR",
) -> str:
    """Turn one frame dict from ``SceneGraphBuilder.update()`` into DOT source.

    Args:
        frame_graph: ``{frame_id, prompt, nodes, edges, ...}``.
        title:       graph caption.  Defaults to frame id + prompt.
        scores:      ``{track_id: weight}`` from ``score_candidates`` — shown as
                     a ``score`` row on the candidate nodes it covers.
        node_notes:  ``{track_id: {label: value}}`` of anything else worth
                     printing per node (e.g. the GT role it matched).
        rankdir:     Graphviz layout direction.
    """
    nodes = frame_graph.get("nodes", [])
    edges = frame_graph.get("edges", [])

    if title is None:
        prompt = frame_graph.get("prompt", "")
        title = f'frame {frame_graph.get("frame_id", "?")}'
        if prompt:
            title += f'   prompt: "{prompt}"'
        title += f"   {len(nodes)} nodes / {len(edges)} edges"

    role_of = {int(n["track_id"]): n.get("role", ROLE_TARGET) for n in nodes}

    lines = [
        "digraph scene_graph {",
        f'  rankdir={rankdir};',
        '  bgcolor="white";',
        f'  label=<<FONT POINT-SIZE="13"><B>{_esc(title)}</B></FONT>>;',
        '  labelloc="t";',
        '  fontname="Helvetica";',
        '  node [shape=box, style="rounded,filled", fontname="Helvetica", penwidth=1.6];',
        '  edge [fontname="Helvetica", fontsize=9];',
    ]

    for node in nodes:
        tid = int(node["track_id"])
        style = _ROLE_STYLE.get(node.get("role", ROLE_TARGET), _UNKNOWN_STYLE)
        extra: Dict[str, Any] = {}
        if node_notes and tid in node_notes:
            extra.update(node_notes[tid])
        if scores is not None and tid in scores:
            extra["score"] = float(scores[tid])
        lines.append(
            f'  n{tid} [label={_node_label(node, extra)}, '
            f'fillcolor="{style["fill"]}", color="{style["border"]}"];'
        )

    for edge in edges:
        src, dst = int(edge["source"]), int(edge["target"])
        rels = edge.get("relations") or []
        # An anchor edge is kept even when empty -- "no relation held this frame"
        # is information.  Draw that case dashed so it reads as an absence.
        spans_anchor = ROLE_ANCHOR in (role_of.get(src), role_of.get(dst))
        if rels:
            label = "\\n".join(_esc(r) for r in rels)
            attrs = f'label="{label}", '
            attrs += ('color="#c8791a", penwidth=1.8, fontcolor="#8a4a00"' if spans_anchor
                      else 'color="#8a8a8a"')
        else:
            attrs = ('label="(no relation)", style=dashed, '
                     'color="#c0c0c0", fontcolor="#a0a0a0"')
        lines.append(f"  n{src} -> n{dst} [{attrs}];")

    lines.append("}")
    return "\n".join(lines)


def query_to_dot(query, *, title: str = "parsed query") -> str:
    """The subgraph the parser asks for: ``(target)-[relation]->(anchor)``.

    The pattern ``score_candidates`` has to match a frame graph against — worth
    rendering next to the scene graph, since a relation that appears on no edge
    is exactly why a candidate scores flat.
    """
    def _ent(entity, role_label, fill, border):
        attrs = ", ".join(f"{k}={v}" for k, v in (entity.get("attrs") or {}).items() if v)
        rows = [f'<TR><TD BGCOLOR="{border}"><FONT COLOR="white"><B>'
                f'{_esc(role_label)}</B></FONT></TD></TR>',
                f'<TR><TD><B>{_esc(entity["class"])}</B></TD></TR>']
        if attrs:
            rows.append(f'<TR><TD><FONT POINT-SIZE="9">{_esc(attrs)}</FONT></TD></TR>')
        label = ('<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="1" CELLPADDING="2">'
                 + "".join(rows) + "</TABLE>>")
        return f'label={label}, fillcolor="{fill}", color="{border}"'

    q = query.to_dict() if hasattr(query, "to_dict") else query
    lines = [
        "digraph query {",
        "  rankdir=LR;",
        '  bgcolor="white";',
        f'  label=<<FONT POINT-SIZE="13"><B>{_esc(title)}</B></FONT>>;',
        '  labelloc="t";',
        '  node [shape=box, style="rounded,filled", fontname="Helvetica", penwidth=1.6];',
        '  edge [fontname="Helvetica", fontsize=10];',
        f'  target [{_ent(q["target"], "target", "#e8f4ff", "#2f6fb0")}];',
    ]
    rel = q.get("relation")
    if q.get("anchor"):
        lines.append(f'  anchor [{_ent(q["anchor"], "anchor", "#fff1de", "#c8791a")}];')
        name = rel["name"] if rel else "?"
        lines.append(f'  target -> anchor [label="{_esc(name)}", penwidth=1.8, '
                     f'color="#c8791a", fontcolor="#8a4a00"];')
    elif rel:
        # Unary or ego-anchored: no second node exists to point at.
        kind = "unary" if rel["arity"] == "unary" else "ego / no tracked anchor"
        lines.append(f'  ego [label=<<B>{_esc(kind)}</B>>, shape=note, '
                     f'fillcolor="#f4f4f4", color="#999999"];')
        lines.append(f'  target -> ego [label="{_esc(rel["name"])}", style=dashed, '
                     f'color="#999999"];')
    lines.append("}")
    return "\n".join(lines)


def render_dot(dot_src: str, out_path: Optional[str] = None, fmt: str = "svg") -> bytes:
    """Run ``dot`` over DOT source.  Writes ``out_path`` if given; returns bytes."""
    exe = shutil.which("dot")
    if exe is None:
        raise RuntimeError(
            "graphviz `dot` not found on PATH — install it "
            "(conda install graphviz / apt-get install graphviz)"
        )
    proc = subprocess.run([exe, f"-T{fmt}"], input=dot_src.encode(),
                          capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"dot failed:\n{proc.stderr.decode()[:2000]}")
    if out_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        with open(out_path, "wb") as f:
            f.write(proc.stdout)
    return proc.stdout


def show_dot(dot_src: str):
    """Inline SVG for a Jupyter cell (returns the object — display it)."""
    from IPython.display import SVG
    return SVG(render_dot(dot_src, fmt="svg"))


def show_frame_graph(frame_graph: Dict[str, Any], **kwargs):
    """``show_dot(frame_graph_to_dot(...))`` — the common case."""
    return show_dot(frame_graph_to_dot(frame_graph, **kwargs))


# ── CLI ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jsonl", required=True, help="scene graph JSONL from SceneGraphBuilder")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--frames", nargs="*", type=int, default=None,
                    help="0-indexed positions in the JSONL (default: all)")
    ap.add_argument("--format", default="svg", help="dot output format (svg, png, pdf)")
    ap.add_argument("--rankdir", default="LR", choices=["LR", "TB", "RL", "BT"])
    args = ap.parse_args()

    with open(args.jsonl) as f:
        graphs = [json.loads(line) for line in f if line.strip()]
    if args.frames is not None:
        graphs = [graphs[i] for i in args.frames if i < len(graphs)]

    os.makedirs(args.out, exist_ok=True)
    for fg in graphs:
        path = os.path.join(args.out, f"frame_{fg.get('frame_id', 0):06d}.{args.format}")
        render_dot(frame_graph_to_dot(fg, rankdir=args.rankdir), path, fmt=args.format)
        print(f"  {os.path.basename(path)}  "
              f"(nodes={len(fg.get('nodes', []))}, edges={len(fg.get('edges', []))})")
    print(f"\n{len(graphs)} graphs → {args.out}/")


if __name__ == "__main__":
    main()
