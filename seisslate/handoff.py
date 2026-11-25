"""Create or verify a portable local review handoff."""

from argparse import ArgumentParser

from .review import ReviewHistory, Role
from .session import SessionSourceError, load_handoff, write_handoff_manifest
from .sources import SourceCatalog


def _verify_paired(catalog: SourceCatalog, history: ReviewHistory) -> int:
    state = history.current.state
    decisions = {decision.target: decision for decision in state.pick_decisions}
    roles: tuple[Role, ...] = ("original", "calibration")
    failed = False
    for blow in state.blows:
        for role in roles:
            decision = decisions.get((blow.blow_id, role))
            status = decision.status if decision else "legacy-unassessed"
            replay = history.replay_pick_evidence(catalog, blow.blow_id, role)
            if replay.raw_samples == "not_recorded":
                print(f"{blow.blow_id} {role} decision {status}, raw not_recorded, attestation unassessed, ready false")
                continue
            verified = (
                replay.source_identity == "source_match" and replay.raw_samples == "raw_samples_match"
                and replay.grid_matches is True and replay.recorded_view_valid is True
            )
            failed |= not verified
            attestation = "unverified"
            if replay.attestation_current:
                attestation = "current"
            elif replay.masks_changed:
                attestation = "stale"
            ready = status == "accepted" and replay.attestation_current
            print(
                f"{blow.blow_id} {role} decision {status}, source {replay.source_identity}, raw {replay.raw_samples}, "
                f"grid match {replay.grid_matches}, recorded view valid {replay.recorded_view_valid}, "
                f"attestation {attestation}, ready {str(ready).lower()}"
            )
    return int(failed)


def main() -> int:
    parser = ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create", help="write an index of existing files")
    create.add_argument("manifest")
    create.add_argument("session")
    create.add_argument("--log", action="append", default=[], help="explicitly include one retained log or evidence file")
    verify = commands.add_parser("verify", help="verify listed files and replay the applied revision")
    verify.add_argument("manifest")
    args = parser.parse_args()
    try:
        if args.command == "create":
            write_handoff_manifest(args.manifest, args.session, logs=args.log)
            print(f"Wrote {args.manifest}")
            return 0
        catalog, history = load_handoff(args.manifest)
        revision = history.current
        print(f"Verified {len(catalog.sources)} recording(s), applied revision {revision.revision_id} ({revision.state.mode})")
        if revision.state.rms is not None:
            replay = history.replay_rms(catalog)
            print(f"RMS replay {replay.status}, disposition {replay.disposition}, ready {str(replay.ready).lower()}")
            if replay.current is not None:
                current = replay.current
                print(f"Current epoch {current.epoch_status}, units {current.units}")
                if current.reasons:
                    print(f"Unavailable reasons {', '.join(current.reasons)}")
            return int(replay.status in ("unsupported_method", "numeric_mismatch"))
        return _verify_paired(catalog, history)
    except SessionSourceError as error:
        parser.exit(1, f"{error} ({error.path.name}, SHA-256 {error.expected_hash})\n")
    except (OSError, ValueError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
