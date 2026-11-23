"""Open the shipped synthetic recordings and their original review baseline."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from .analysis import in_strict_lag_window
from .review import Blow, Receiver, ReviewHistory, ReviewState, import_pick
from .sources import SourceCatalog, load_source, utc_ns


def load_example(directory: Path | str | None = None) -> tuple[SourceCatalog, ReviewHistory]:
    """Load the verified fixture and initialize review membership from imported lag."""
    fixture = (
        Path(__file__).resolve().parents[1] / "fixtures" / "synthetic"
        if directory is None
        else Path(directory)
    )
    manifest = json.loads((fixture / "expected.json").read_text(encoding="utf-8"))
    sources = []
    receivers = []
    for role in ("original", "calibration"):
        entry = manifest["sources"][role]
        source = load_source(fixture / entry["path"])
        if source.sha256 != entry["sha256"] or source.size_bytes != entry["size_bytes"]:
            raise ValueError(f"Synthetic {role} recording does not match its manifest")
        sources.append(source)
        receivers.append(Receiver(role, entry["trace_id"], source.sha256))
    catalog = SourceCatalog(tuple(sources))

    blows = []
    with (fixture / "baseline.csv").open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            picks = []
            for receiver in receivers:
                value = row[f"{receiver.role}_pick_utc"]
                picks.append(import_pick(
                    catalog,
                    receiver.channel,
                    utc_ns(value) if value else None,
                    source_hash=receiver.source_hash,
                ))
            original, calibration = picks
            t1, t2 = original.imported_ns, calibration.imported_ns
            included = t1 is not None and t2 is not None and in_strict_lag_window(t1 - t2)
            if row["strict_included"] != str(included).lower():
                raise ValueError(f"Recorded strict membership disagrees with imported lag for {row['blow_id']}")
            blows.append(Blow(
                row["blow_id"],
                original,
                calibration,
                float(row["distance_original_m"]),
                float(row["distance_calibration_m"]),
                included=included,
                reason="" if included else row["case"],
            ))
    state = ReviewState(
        blows=tuple(blows),
        receivers=tuple(receivers),
        timing_evidence="Synthetic recordings share a known integer-millisecond clock.",
    )
    return catalog, ReviewHistory(state)
