#!/usr/bin/env python3
"""
Run the query parser over the ANSR benchmark episode prompts and print a
human-readable report.

Reads the ``task`` string out of every ``description.json`` in an episode
release — straight out of the release ``.zip`` files, no unpacking needed —
groups identical prompts, parses each one, and renders the resulting ``Query``
in plain English next to its structured form so a mis-parse is obvious at a
glance.

Usage:
    python parse_benchmark_prompts.py                        # all releases
    python parse_benchmark_prompts.py --release v3           # one release
    python parse_benchmark_prompts.py --all                  # per-episode, no grouping
    python parse_benchmark_prompts.py --failures             # only prompts that failed
    python parse_benchmark_prompts.py --prompt "red car behind the bus"
    python parse_benchmark_prompts.py --out report.txt
"""

from __future__ import annotations

import argparse
import collections
import io
import json
import os
import re
import sys
import zipfile
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from query_parser import (  # noqa: E402
    COLOR_SET,
    RELATIONS,
    QueryParseError,
    parse,
)

DEFAULT_EPISODES = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "example_benchmark_episodes",
)

WIDTH = 88
RULE = "─" * WIDTH
HEAVY = "═" * WIDTH


# ---------------------------------------------------------------------------
# Loading prompts
# ---------------------------------------------------------------------------

@dataclass
class Episode:
    """One benchmark episode's task prompt plus where it came from."""

    prompt: str
    release: str          # 'episode-release-v3'
    bundle: str           # 'tracking-0'
    episode_id: str       # 'episode-000'

    @property
    def source(self) -> str:
        return f"{self.release}/{self.bundle}/{self.episode_id}"


def _episode_from_description(raw: bytes, release: str, bundle: str) -> Optional[Episode]:
    try:
        data = json.loads(raw)
    except (ValueError, UnicodeDecodeError):
        return None
    task = (data.get("scenario_objective") or {}).get("task")
    if not task:
        return None
    return Episode(
        prompt=" ".join(task.split()),  # collapse the double spaces some prompts carry
        release=release,
        bundle=bundle,
        episode_id=data.get("scenario_id", "?"),
    )


def load_episodes(root: str, release_filter: Optional[str] = None) -> List[Episode]:
    """Collect episodes from a checkout of ``example_benchmark_episodes``.

    Handles both the shipped ``.zip`` bundles and already-extracted directories.
    """
    episodes: List[Episode] = []
    seen: set = set()

    for dirpath, _dirnames, filenames in os.walk(root):
        rel = os.path.relpath(dirpath, root)
        parts = [p for p in rel.split(os.sep) if p not in (".", "")]
        if parts and parts[0].startswith("."):
            continue
        release = parts[0] if parts else os.path.basename(root)
        if release_filter and release_filter not in release:
            continue

        for name in sorted(filenames):
            path = os.path.join(dirpath, name)

            if name == "description.json":
                bundle = parts[1] if len(parts) > 1 else "-"
                with open(path, "rb") as fh:
                    ep = _episode_from_description(fh.read(), release, bundle)
                if ep and ep.source not in seen:
                    seen.add(ep.source)
                    episodes.append(ep)

            elif name.endswith(".zip"):
                bundle = os.path.splitext(name)[0]
                try:
                    zf = zipfile.ZipFile(path)
                except zipfile.BadZipFile:
                    continue
                with zf:
                    for member in sorted(zf.namelist()):
                        if not member.endswith("description.json"):
                            continue
                        if member.startswith("__MACOSX/"):
                            continue
                        ep = _episode_from_description(zf.read(member), release, bundle)
                        if ep and ep.source not in seen:
                            seen.add(ep.source)
                            episodes.append(ep)

    episodes.sort(key=lambda e: e.source)
    return episodes


# ---------------------------------------------------------------------------
# Rendering one parse
# ---------------------------------------------------------------------------

def _describe_entity(entity: Optional[Dict]) -> str:
    """'red car (size=large)' — the entity as a person would say it."""
    if entity is None:
        return "—"
    attrs = dict(entity.get("attrs") or {})
    lead: List[str] = []
    for key in ("count", "color", "size"):
        if key in attrs:
            lead.append(str(attrs.pop(key)))
    for extra in attrs.pop("other", []):
        lead.append(str(extra))
    text = " ".join(lead + [entity.get("class", "?")])
    if attrs:
        text += " (" + ", ".join(f"{k}={v}" for k, v in sorted(attrs.items())) + ")"
    return text


#: Canonical relation name -> how to say it in a sentence.
_RELATION_PHRASE = {
    "behind": "behind",
    "in_front_of": "in front of",
    "left_of": "to the left of",
    "right_of": "to the right of",
    "next_to": "next to",
    "between": "between",
    "moving": "that is moving",
    "stationary": "that is stationary",
    "braking": "that is braking",
    "turning": "that is turning",
    "counter_direction": "travelling in the counter direction",
    "same_direction": "travelling in the same direction",
    "following": "following",
    "approaching": "approaching",
    "overtaking": "overtaking",
    "leading": "leading",
}


def _reads_as(query) -> str:
    """One-line plain-English read-back of the structured query."""
    target = _describe_entity(query.target)
    if query.relation is None:
        return f'track every "{target}"  (plain MOT — no anchor, no relation)'

    name = query.relation["name"]
    phrase = _RELATION_PHRASE.get(name, name)
    if query.relation["arity"] == "unary":
        return f'track every "{target}" {phrase}'
    if query.anchor is None:
        return f'track every "{target}" {phrase} <the ego vehicle>'
    return f'track every "{target}" {phrase} the "{_describe_entity(query.anchor)}"'


def _wrap(text: str, indent: str, width: int = WIDTH) -> str:
    import textwrap

    return textwrap.fill(
        text, width=width, initial_indent=indent, subsequent_indent=indent + "  "
    )


def _sanity_flags(query) -> List[str]:
    """Cheap heuristics that catch a parse that is structurally valid but wrong.

    These are report-only; the parser itself makes no such judgement.
    """
    flags: List[str] = []
    cls = (query.target or {}).get("class", "")
    if cls in COLOR_SET:
        flags.append(f'target class is the colour word "{cls}" — the head noun was lost')
    if query.anchor and query.anchor.get("class") in {"time", "second", "sec", "interval"}:
        flags.append(
            f'anchor is "{query.anchor["class"]}" — a time window was read as a spatial object'
        )
    if cls in {"number", "vehicle", "thing", "one", "type"} and not (query.target or {}).get("attrs"):
        flags.append(f'target class "{cls}" carries no discriminative signal')
    return flags


def render_group(
    index: int,
    prompt: str,
    episodes: List[Episode],
    show_sources: int = 3,
) -> Tuple[str, str, List[str]]:
    """Render one prompt group.  Returns (text, status, sanity_flags)."""
    out = io.StringIO()
    n = len(episodes)
    releases = sorted({e.release.replace("episode-release-", "") for e in episodes})

    out.write(RULE + "\n")
    out.write(f"[{index}] {n} episode{'s' if n != 1 else ''}   ·   release {', '.join(releases)}\n")
    out.write(_wrap(f'PROMPT  "{prompt}"', "  ") + "\n")
    shown = [e.source for e in episodes[:show_sources]]
    more = f"  (+{n - len(shown)} more)" if n > len(shown) else ""
    out.write(f"  from    {', '.join(shown)}{more}\n\n")

    try:
        query = parse(prompt)
    except QueryParseError as exc:
        status = type(exc).__name__
        out.write(f"  RESULT  ✗ {status}\n")
        out.write(_wrap(str(exc).split(" Add it to")[0].strip(), "          ") + "\n")
        return out.getvalue(), status, []

    flags = _sanity_flags(query)
    status = "ok"
    out.write("  RESULT  ✓ parsed\n")
    out.write(_wrap("reads as:  " + _reads_as(query), "          ") + "\n\n")

    out.write(f"    target    {_describe_entity(query.target)}\n")
    out.write(f"              class={query.target['class']!r}  attrs={query.target['attrs']}\n")

    if query.relation is None:
        out.write("    relation  — none (classic MOT)\n")
        out.write("    anchor    — none\n")
    else:
        rel = query.relation
        spec = RELATIONS[rel["name"]]
        out.write(
            f"    relation  {rel['name']}  ·  {spec.kind}  ·  {rel['arity']}"
            f"  ·  temporal={rel['temporal']}  ·  weight={rel['weight']:.2f}\n"
        )
        if query.anchor is None:
            reason = "unary relation" if rel["arity"] == "unary" else "ego-anchored"
            out.write(f"    anchor    — none ({reason})\n")
        else:
            out.write(f"    anchor    {_describe_entity(query.anchor)}\n")
            out.write(
                f"              class={query.anchor['class']!r}  attrs={query.anchor['attrs']}\n"
            )

    for note in query.notes:
        out.write(_wrap("note   " + note, "    ") + "\n")
    for flag in flags:
        out.write(_wrap("⚠ flag  " + flag, "    ") + "\n")

    return out.getvalue(), status, flags


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _prompt_family(prompt: str) -> str:
    """Coarse bucket for the summary — numbers and street names stripped out."""
    low = prompt.lower()
    if low.startswith("find and track"):
        return "tracking:  'Find and track the <colour> <TYPE> between times a and b'"
    if low.startswith("count the number of vehicles that turn"):
        return "counting:  'Count the number of vehicles that turn at <intersection> in [a, b]'"
    return "other:     free-form v0 mission prompts"


def render_summary(rows: List[Tuple[str, List[Episode], str, List[str]]]) -> str:
    out = io.StringIO()
    total_prompts = len(rows)
    total_eps = sum(len(eps) for _p, eps, _s, _f in rows)

    out.write("\n" + HEAVY + "\n")
    out.write("SUMMARY\n")
    out.write(HEAVY + "\n\n")
    out.write(f"  {total_eps} episodes  ->  {total_prompts} distinct prompts\n\n")

    by_status = collections.Counter(status for _p, _e, status, _f in rows)
    eps_by_status = collections.Counter()
    for _p, eps, status, _f in rows:
        eps_by_status[status] += len(eps)

    out.write("  Outcome                       prompts    episodes\n")
    out.write("  " + "-" * 52 + "\n")
    for status, count in by_status.most_common():
        mark = "✓" if status == "ok" else "✗"
        out.write(f"  {mark} {status:<26} {count:>7} {eps_by_status[status]:>11}\n")

    flagged = [r for r in rows if r[2] == "ok" and r[3]]
    flagged_eps = sum(len(r[1]) for r in flagged)
    clean = [r for r in rows if r[2] == "ok" and not r[3]]
    clean_eps = sum(len(r[1]) for r in clean)
    out.write("\n  Of the parses that succeeded:\n")
    out.write(f"    clean                      {len(clean):>7} {clean_eps:>11}\n")
    out.write(f"    flagged as suspect         {len(flagged):>7} {flagged_eps:>11}\n")

    out.write("\n  By prompt family\n")
    out.write("  " + "-" * 52 + "\n")
    fam = collections.defaultdict(lambda: collections.Counter())
    fam_eps = collections.Counter()
    for prompt, eps, status, flags in rows:
        key = _prompt_family(prompt)
        bucket = "ok" if status == "ok" and not flags else ("suspect" if status == "ok" else status)
        fam[key][bucket] += len(eps)
        fam_eps[key] += len(eps)
    for key, counts in sorted(fam.items(), key=lambda kv: -fam_eps[kv[0]]):
        out.write(f"  {key}\n")
        for bucket, count in counts.most_common():
            out.write(f"      {bucket:<28} {count:>5} episodes\n")

    rel_hist = collections.Counter()
    cls_hist = collections.Counter()
    for prompt, eps, status, _flags in rows:
        if status != "ok":
            continue
        try:
            q = parse(prompt)
        except QueryParseError:
            continue
        rel_hist[q.relation["name"] if q.relation else "(none)"] += len(eps)
        cls_hist[q.target["class"]] += len(eps)

    if rel_hist:
        out.write("\n  Relations emitted (by episode)\n")
        out.write("  " + "-" * 52 + "\n")
        for name, count in rel_hist.most_common():
            out.write(f"      {name:<28} {count:>5}\n")

    if cls_hist:
        out.write("\n  Target classes emitted (by episode, top 10)\n")
        out.write("  " + "-" * 52 + "\n")
        for name, count in cls_hist.most_common(10):
            out.write(f"      {name:<28} {count:>5}\n")

    flag_hist = collections.Counter()
    for _p, eps, status, flags in rows:
        for flag in flags:
            flag_hist[re.sub(r'"[^"]*"', "<...>", flag)] += len(eps)
    if flag_hist:
        out.write("\n  Sanity flags raised (by episode)\n")
        out.write("  " + "-" * 52 + "\n")
        for flag, count in flag_hist.most_common():
            out.write(_wrap(f"{count:>5}  {flag}", "      ") + "\n")

    return out.getvalue()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", default=DEFAULT_EPISODES,
                    help="path to the example_benchmark_episodes checkout")
    ap.add_argument("--release", default=None,
                    help="only this release, e.g. 'v3'")
    ap.add_argument("--prompt", action="append", default=None,
                    help="parse this prompt instead of the benchmark (repeatable)")
    ap.add_argument("--all", action="store_true",
                    help="one entry per episode instead of one per distinct prompt")
    ap.add_argument("--failures", action="store_true",
                    help="show only prompts that failed or were flagged")
    ap.add_argument("--limit", type=int, default=None,
                    help="stop after N entries")
    ap.add_argument("--summary-only", action="store_true", help="skip the per-prompt detail")
    ap.add_argument("--json", dest="as_json", action="store_true",
                    help="emit machine-readable JSON instead of the report")
    ap.add_argument("--out", default=None, help="also write the report to this file")
    args = ap.parse_args(argv)

    if args.prompt:
        episodes = [Episode(p, "adhoc", "-", f"prompt-{i:03d}")
                    for i, p in enumerate(args.prompt)]
    else:
        if not os.path.isdir(args.episodes):
            ap.error(f"episode directory not found: {args.episodes}")
        episodes = load_episodes(args.episodes, args.release)
        if not episodes:
            ap.error(f"no episode description.json found under {args.episodes}")

    groups: "collections.OrderedDict[str, List[Episode]]" = collections.OrderedDict()
    if args.all:
        # Key on the source so identical prompts stay separate entries.
        for ep in episodes:
            groups[f"{ep.source}\x00{ep.prompt}"] = [ep]
    else:
        for ep in episodes:
            groups.setdefault(ep.prompt, []).append(ep)
        groups = collections.OrderedDict(
            sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0]))
        )

    rows: List[Tuple[str, List[Episode], str, List[str]]] = []
    blocks: List[str] = []
    for i, (key, eps) in enumerate(groups.items(), start=1):
        prompt = key.split("\x00", 1)[1] if args.all else key
        text, status, flags = render_group(i, prompt, eps)
        rows.append((prompt, eps, status, flags))
        # --limit trims the printed detail only; the summary always covers everything.
        if args.failures and status == "ok" and not flags:
            continue
        if args.limit and len(blocks) >= args.limit:
            continue
        blocks.append(text)

    if args.as_json:
        payload = []
        for prompt, eps, status, flags in rows:
            entry = {
                "prompt": prompt,
                "episodes": [e.source for e in eps],
                "episode_count": len(eps),
                "status": status,
                "sanity_flags": flags,
            }
            if status == "ok":
                q = parse(prompt)
                entry["query"] = q.to_dict()
                entry["notes"] = q.notes
                entry["reads_as"] = _reads_as(q)
            payload.append(entry)
        text = json.dumps(payload, indent=2)
    else:
        header = (
            HEAVY + "\n"
            "QUERY PARSER  ·  ANSR benchmark episode prompts\n"
            f"episodes: {args.episodes}\n"
            + HEAVY + "\n\n"
        )
        body = "" if args.summary_only else "\n".join(blocks)
        text = header + body + render_summary(rows)

    sys.stdout.write(text + "\n")
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(text + "\n")
        sys.stderr.write(f"\nwrote {args.out}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
