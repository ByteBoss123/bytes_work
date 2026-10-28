"""Stream a (gzipped) XES event log into flat case and event tables.

Usage: python -I src/parse_xes.py <log.xes.gz> <out_dir>
Writes events.parquet (one row per event, trace attributes prefixed `case:`)
and cases.parquet (one row per trace).
"""
from __future__ import annotations

import gzip
import sys
from pathlib import Path
from xml.etree.ElementTree import iterparse

import pandas as pd

CAST = {"string": str, "int": int, "float": float, "boolean": lambda v: v.lower() == "true",
        "date": lambda v: v, "id": str}


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def parse(path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    opener = gzip.open if path.endswith(".gz") else open
    events, cases = [], []
    trace_attrs: dict | None = None
    event_attrs: dict | None = None
    depth = 0
    with opener(path, "rb") as fh:
        for kind, el in iterparse(fh, events=("start", "end")):
            tag = _local(el.tag)
            if kind == "start":
                depth += 1
                if tag == "trace":
                    trace_attrs = {}
                elif tag == "event" and trace_attrs is not None:
                    event_attrs = {}
                continue
            depth -= 1
            if tag in CAST and "key" in el.attrib:
                val = CAST[tag](el.attrib.get("value", ""))
                if event_attrs is not None:
                    event_attrs[el.attrib["key"]] = val
                elif trace_attrs is not None:
                    trace_attrs[el.attrib["key"]] = val
                el.clear()
            elif tag == "event" and event_attrs is not None:
                events.append({**{f"case:{k}": v for k, v in trace_attrs.items()}, **event_attrs})
                event_attrs = None
                el.clear()
            elif tag == "trace" and trace_attrs is not None:
                cases.append(trace_attrs)
                trace_attrs = None
                el.clear()
    ev = pd.DataFrame(events)
    if "time:timestamp" in ev:
        ev["time:timestamp"] = pd.to_datetime(ev["time:timestamp"], utc=True, format="ISO8601")
    return ev, pd.DataFrame(cases)


if __name__ == "__main__":
    src, out = sys.argv[1], Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    ev, cs = parse(src)
    ev.to_parquet(out / "events.parquet", index=False)
    cs.to_parquet(out / "cases.parquet", index=False)
    print(f"cases={len(cs):,} events={len(ev):,} columns={list(ev.columns)}")
