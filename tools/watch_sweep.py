#!/usr/bin/env python3
"""Automated watch sweep for the 70B-on-16GB research track.

Runs the recurring release/upload watches (HuggingFace + GitHub, read-only
API metadata — no model files are downloaded) that previously required
manual sweeps, and prints a DELTA report against the previous snapshot so a
human only reads what changed.

Checks:
  1. tq_gguf_70b     — TQ1_0 / TQ2_0 top-50-by-downloads: any >= 60B params?
  2. jangqai_latest  — JANGQ-AI org newest uploads (JANG_2L/JANG_2M 35B-A3B watch)
  3. jundot_latest   — Jundot org newest uploads (oQ ecosystem health)
  4. ddalcu_iqmlx    — ddalcu org: Qwen3.6-35B-A3B-iQ-MLX watch
  5. nemotron_dl     — Nemotron-3-Nano-Omni-30B-A3B-JANGTQ2 download count
  6. microsoft_org   — microsoft org recent uploads (BitNet ternary watch)
  7. vmlx_release    — latest vmlx GitHub release (jjang-ai/vmlx)
  8. mlxl3_release   — latest mlxl3 GitHub release (0xZKnw/mlxl3)
  9. exl3_35b        — newest 35B-A3B EXL3 uploads (2-bpw-class watch)
  10. novamlx_release — latest novamlx GitHub release (cnshsliu/novamlx;
      TIE/Smelt paging + 2-bit format support are the watch conditions)
  11. vmlx_oq_loader — whether vmlx main ships an oQ-format loader (the
      dormant oQ2 rung-4 path: Jundot's oQ family has the best measured
      ~2-bit 35B datapoint but vmlx can't serve it)

Usage:
    python3 -m tools.watch_sweep [--snapshot PATH] [--full]

The snapshot (default research/data/watch_snapshot.json, gitignored) records
the last sweep; deltas are computed against it. --full prints the complete
baseline alongside the deltas (useful for the log).

Exit code: 0 if all checks fetched, 1 if any check failed to fetch (the
partial report is still printed so the rest of the sweep is usable).

One fetch per check, no retries — the sweeps run every 30 minutes anyway.
"""

import argparse
import datetime
import json
import os
import re
import sys
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_HF = "https://huggingface.co/api/models"
_GH = "https://api.github.com/repos"
_DEFAULT_TIMEOUT = 20

# Param-size heuristic for the >=60B filter: largest "<N>B" token in the
# repo id (e.g. "Qwen3.5-397B-A17B-TQ1_0-GGUF" -> 397; "qwen3.5-35b-a3b"
# -> 35, the "A3B" active-params token loses to the total). Consistent with
# the manual sweeps, which flagged the 397B oddities on id naming.
_PARAM_RE = re.compile(r"(\d+(?:\.\d+)?)\s*B(?![a-zA-Z])", re.IGNORECASE)

_BIG_PARAM_BAR = 60.0


def http_get(url, timeout=_DEFAULT_TIMEOUT):
    """Fetch a URL and decode JSON. Raises WatchError on any failure."""
    req = urllib.request.Request(url, headers={"User-Agent": "hearth-watch-sweep/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.load(resp)
    except (urllib.error.URLError, OSError, ValueError, json.JSONDecodeError) as e:
        raise WatchError("%s: %s" % (url, e))


class WatchError(Exception):
    """One check could not fetch its source."""


def _params_billions(repo_id):
    """Estimated total params (billions) from the repo id, or None."""
    hits = _PARAM_RE.findall(repo_id or "")
    if not hits:
        return None
    return max(float(h) for h in hits)


def _hf_models(params, get):
    return get(_HF + params)


def _top_n(entries, n=3):
    return [
        {
            "id": e.get("id"),
            "downloads": e.get("downloads"),
            "lastModified": e.get("lastModified"),
            "paramsB": _params_billions(e.get("id")),
        }
        for e in (entries or [])[:n]
    ]


def check_tq_gguf_70b(get):
    """TQ1_0/TQ2_0 top-50-by-downloads: any entry estimated >= 60B params."""
    big = []
    tops = {}
    for quant in ("TQ1_0", "TQ2_0"):
        entries = _hf_models(
            "?search=%s&sort=downloads&direction=-1&limit=50" % quant, get
        )
        tops[quant] = entries[0]["id"] if entries else None
        for e in entries:
            pb = _params_billions(e.get("id"))
            if pb is not None and pb >= _BIG_PARAM_BAR:
                big.append(
                    {
                        "quant": quant,
                        "id": e.get("id"),
                        "downloads": e.get("downloads"),
                        "paramsB": pb,
                    }
                )
    return {"top": tops, "big_entries": big}


def check_jangqai_latest(get):
    """JANGQ-AI org: newest uploads (JANG_2L/JANG_2M 35B-A3B watch)."""
    entries = _hf_models(
        "?author=JANGQ-AI&sort=lastModified&direction=-1&limit=10", get
    )
    jang_2x = [
        e.get("id")
        for e in entries
        if re.search(r"35B-A3B.*JANG_2[LMB]", e.get("id") or "", re.IGNORECASE)
    ]
    return {
        "newest": _top_n(entries, 3),
        "jang_2x_35b": jang_2x,
        "newest_modified": entries[0].get("lastModified") if entries else None,
    }


def check_jundot_latest(get):
    """Jundot org: newest uploads (oQ ecosystem health)."""
    entries = _hf_models(
        "?author=Jundot&sort=lastModified&direction=-1&limit=10", get
    )
    oq_35b = [
        e.get("id")
        for e in entries
        if re.search(r"35B-A3B", e.get("id") or "", re.IGNORECASE)
        and re.search(r"oQ", e.get("id") or "")
    ]
    return {
        "newest": _top_n(entries, 3),
        "oq_35b": oq_35b,
        "newest_modified": entries[0].get("lastModified") if entries else None,
    }


def check_ddalcu_iqmlx(get):
    """ddalcu org: is there a Qwen3.6-35B-A3B-iQ-MLX upload yet?"""
    entries = _hf_models("?author=ddalcu&sort=lastModified&direction=-1&limit=30", get)
    hits = [
        {"id": e.get("id"), "lastModified": e.get("lastModified")}
        for e in entries
        if re.search(r"Qwen3\.?6-35B-A3B", e.get("id") or "", re.IGNORECASE)
    ]
    return {
        "hits": hits,
        "family_top": max(
            (
                (e.get("id"), e.get("downloads"))
                for e in entries
                if "iQ-MLX" in (e.get("id") or "")
            ),
            default=None,
            key=lambda t: t[1] or 0,
        ),
        "newest_modified": entries[0].get("lastModified") if entries else None,
    }


def check_nemotron_dl(get):
    """Nemotron-3-Nano-Omni-30B-A3B-JANGTQ2 download count (quality-signal watch)."""
    meta = get(_HF + "/JANGQ-AI/Nemotron-3-Nano-Omni-30B-A3B-JANGTQ2")
    return {"downloads": meta.get("downloads"), "likes": meta.get("likes")}


def check_microsoft_org(get):
    """microsoft org recent uploads (BitNet ternary-model watch)."""
    entries = _hf_models(
        "?author=microsoft&sort=lastModified&direction=-1&limit=20", get
    )
    bitnet = [
        {"id": e.get("id"), "downloads": e.get("downloads")}
        for e in entries
        if re.search(r"bitnet", e.get("id") or "", re.IGNORECASE)
    ]
    largest = None
    for e in bitnet:
        pb = _params_billions(e.get("id"))
        if pb is not None and (largest is None or pb > largest["paramsB"]):
            largest = {"id": e.get("id"), "paramsB": pb}
    return {
        "newest_id": entries[0].get("id") if entries else None,
        "newest_modified": entries[0].get("lastModified") if entries else None,
        "recent_bitnet": bitnet[:5],
        "largest_recent_bitnet": largest,
    }


def _gh_latest_release(repo, get):
    releases = get("%s/%s/releases?per_page=3" % (_GH, repo))
    if not releases:
        return {"tag": None, "published_at": None}
    r = releases[0]
    return {"tag": r.get("tag_name"), "published_at": r.get("published_at")}


def check_vmlx_release(get):
    return {"release": _gh_latest_release("jjang-ai/vmlx", get)}


def check_mlxl3_release(get):
    return {"release": _gh_latest_release("0xZKnw/mlxl3", get)}


def check_novamlx_release(get):
    """Latest novamlx GitHub release.

    novamlx (cnshsliu/novamlx) is the Mac-native Swift MoE-paging server
    with NovaMLX-TIE 3-tier streaming — architecturally a smarter Smelt for
    the 13.1 GB oQ2. The watch conditions from the 2026-09-22 doc check:
    (a) 2-bit format support landing (currently SafeTensors 4/8-bit, FP16,
    NVFP4 only — cannot serve oQ2), (b) a 4-bit 35B-A3B path worth
    comparing against vmlx --smelt. Both would arrive as release-note
    headlines, which is what this check tracks.
    """
    return {"release": _gh_latest_release("cnshsliu/novamlx", get)}


def check_vmlx_oq_loader(get):
    """Whether vmlx ships an oQ-format loader (the dormant oQ2 rung-4 path).

    Jundot's oQ family (oQ2/oQ3e/oQ4e, incl. DeepSeek-V4.1-Flash at 2-3.8k
    downloads) is the healthiest ~2-bit MoE ecosystem on HF with measured
    quality datapoints (oQ2 64% MMLU on the 3.5 variant), but the rung-4
    oQ2 plan is dormant because vmlx's loaders are jang / jangtq /
    laguna / mistral3 / zaya / qwen4_exp / dsv4 — no oQ. If a loader file
    mentioning 'oq' appears, the dormant oQ2 path is worth reviving.
    """
    entries = get(
        "%s/jjang-ai/vmlx/contents/vmlx_engine/loaders?ref=main" % _GH
    )
    if not isinstance(entries, list):
        raise WatchError("unexpected contents payload: %r" % (entries,))
    names = sorted(
        e.get("name")
        for e in entries
        if isinstance(e, dict) and e.get("type") == "file" and e.get("name")
    )
    return {
        "loader_files": names,
        "oq_loader_present": any("oq" in n.lower() for n in names),
    }


def check_exl3_35b(get):
    """Newest 35B-A3B EXL3 uploads (2-bpw-class watch)."""
    entries = _hf_models(
        "?search=35B-A3B-exl3&sort=lastModified&direction=-1&limit=10", get
    )
    two_bpw = [
        {"id": e.get("id"), "downloads": e.get("downloads")}
        for e in entries
        if re.search(r"2\.0", e.get("id") or "")
    ]
    return {
        "newest": _top_n(entries, 3),
        "two_bpw_hits": two_bpw,
        "newest_modified": entries[0].get("lastModified") if entries else None,
    }


CHECKS = [
    ("tq_gguf_70b", check_tq_gguf_70b),
    ("jangqai_latest", check_jangqai_latest),
    ("jundot_latest", check_jundot_latest),
    ("ddalcu_iqmlx", check_ddalcu_iqmlx),
    ("nemotron_dl", check_nemotron_dl),
    ("microsoft_org", check_microsoft_org),
    ("vmlx_release", check_vmlx_release),
    ("mlxl3_release", check_mlxl3_release),
    ("exl3_35b", check_exl3_35b),
    ("novamlx_release", check_novamlx_release),
    ("vmlx_oq_loader", check_vmlx_oq_loader),
]


def _ids_of(data):
    """Extract the set of repo ids / release tags that identify this check's result."""
    ids = set()
    if isinstance(data, dict):
        for key in ("newest", "big_entries", "hits", "recent_bitnet", "two_bpw_hits"):
            for e in data.get(key) or []:
                if isinstance(e, dict) and e.get("id"):
                    ids.add(e["id"])
        top = data.get("top")
        if isinstance(top, dict):
            ids.update(v for v in top.values() if v)
        rel = data.get("release")
        if isinstance(rel, dict) and rel.get("tag"):
            ids.add(rel["tag"])
        for key in ("newest_id",):
            if data.get(key):
                ids.add(data[key])
        for n in data.get("loader_files") or []:
            if isinstance(n, str):
                ids.add(n)
    return ids


def diff_check(name, old, new):
    """One-line-per-change delta between two snapshots of one check.

    Returns (changed: bool, lines: list[str]).
    """
    lines = []
    if old is None:
        return True, ["%s: baseline recorded" % name]

    old_ids, new_ids = _ids_of(old), _ids_of(new)
    for i in sorted(new_ids - old_ids):
        lines.append("%s: NEW %s" % (name, i))
    for i in sorted(old_ids - new_ids):
        lines.append("%s: GONE %s" % (name, i))

    # Scalar values worth tracking even when the id set is unchanged.
    def scalar(path):
        d_old, d_new = old, new
        for p in path:
            d_old = (d_old or {}).get(p)
            d_new = (d_new or {}).get(p)
        return d_old, d_new

    for path, label in (
        (["downloads"], "nemotron downloads"),
        (["release", "tag"], "release tag"),
        (["newest_modified"], "newest upload"),
        (["oq_loader_present"], "oQ loader"),
    ):
        v_old, v_new = scalar(path)
        if v_old != v_new and v_old is not None:
            lines.append(
                "%s: %s %s -> %s" % (name, label, v_old, v_new)
            )
    return (len(lines) > 0), lines


def run_sweep(get=http_get):
    """Run every check; return (results dict, errors dict)."""
    results, errors = {}, {}
    for name, fn in CHECKS:
        try:
            results[name] = fn(get)
        except WatchError as e:
            errors[name] = str(e)
    return results, errors


def load_snapshot(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def save_snapshot(path, results):
    with open(path, "w") as f:
        json.dump(
            {
                "swept_at": datetime.datetime.now(
                    datetime.timezone.utc
                ).isoformat(timespec="seconds"),
                "checks": results,
            },
            f,
            indent=2,
        )


def format_report(results, errors, old_snapshot, full=False):
    """Render the sweep: deltas first, then a compact baseline summary."""
    out = []
    out.append("WATCH SWEEP %s" % datetime.datetime.now(
        datetime.timezone(datetime.timedelta(hours=-7))
    ).strftime("%Y-%m-%d ~%H:%M PDT"))

    if old_snapshot is not None:
        out.append("Previous sweep: %s" % old_snapshot.get("swept_at"))
    else:
        out.append("Previous sweep: none (first run)")

    out.append("")
    out.append("CHANGES:")
    any_change = False
    old_checks = (old_snapshot or {}).get("checks") or {}
    for name, _ in CHECKS:
        if name in errors:
            continue
        changed, lines = diff_check(name, old_checks.get(name), results[name])
        if changed:
            any_change = True
            out.extend("  * " + l for l in lines)
    if not any_change:
        out.append("  (none)")

    if errors:
        out.append("")
        out.append("ERRORS (manual re-check needed):")
        for name, msg in errors.items():
            out.append("  * %s: %s" % (name, msg))

    if full:
        out.append("")
        out.append("BASELINE:")
        out.extend(format_baseline(results, errors))
    return "\n".join(out)


def format_baseline(results, errors):
    """One compact line per check describing its current state."""
    out = []
    for name, _ in CHECKS:
        if name in errors:
            out.append("  %s: ERROR" % name)
            continue
        if name not in results:
            continue
        d = results[name]
        if name == "tq_gguf_70b":
            big = d["big_entries"]
            out.append(
                "  tq: >=60B entries: %s (top TQ1_0=%s, TQ2_0=%s)"
                % (
                    ", ".join("%s (%dd)" % (b["id"], b["downloads"] or 0) for b in big)
                    or "none",
                    d["top"].get("TQ1_0"),
                    d["top"].get("TQ2_0"),
                )
            )
        elif name in ("jangqai_latest", "jundot_latest", "exl3_35b"):
            out.append(
                "  %s: newest %s"
                % (
                    name,
                    ", ".join(
                        "%s (%s)" % (e["id"], (e["lastModified"] or "?")[:10])
                        for e in d["newest"]
                    ),
                )
            )
        elif name == "ddalcu_iqmlx":
            out.append(
                "  ddalcu_iqmlx: 35B-A3B hits=%s (top family dl: %s)"
                % (
                    ", ".join(
                        "%s (%s)" % (h["id"], (h["lastModified"] or "?")[:10])
                        for h in d["hits"]
                    )
                    or "none",
                    d["family_top"],
                )
            )
        elif name == "nemotron_dl":
            out.append(
                "  nemotron: %s downloads, %s likes"
                % (d["downloads"], d["likes"])
            )
        elif name == "microsoft_org":
            lb = d["largest_recent_bitnet"]
            out.append(
                "  microsoft: newest=%s; largest recent bitnet=%s"
                % (d["newest_id"], lb["id"] if lb else None)
            )
        elif name in ("vmlx_release", "mlxl3_release", "novamlx_release"):
            r = d["release"]
            out.append(
                "  %s: %s (%s)" % (name, r["tag"], (r["published_at"] or "?")[:10])
            )
        elif name == "vmlx_oq_loader":
            out.append(
                "  vmlx_oq_loader: oQ loader %s (loaders: %s)"
                % (
                    "PRESENT" if d["oq_loader_present"] else "absent",
                    ", ".join(d["loader_files"]),
                )
            )
    return out


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--snapshot",
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "research",
            "data",
            "watch_snapshot.json",
        ),
        help="snapshot file for delta computation",
    )
    p.add_argument(
        "--full",
        action="store_true",
        help="print the full baseline as well as the deltas",
    )
    p.add_argument(
        "--timeout",
        type=float,
        default=_DEFAULT_TIMEOUT,
        help="per-request HTTP timeout (seconds)",
    )
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    get = lambda url: http_get(url, timeout=args.timeout)  # noqa: E731
    results, errors = run_sweep(get)
    old = load_snapshot(args.snapshot)
    print(format_report(results, errors, old, full=args.full))
    save_snapshot(args.snapshot, results)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
