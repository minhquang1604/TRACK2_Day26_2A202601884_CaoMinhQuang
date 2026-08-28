"""eval/_drift_data.py — a frozen snapshot of `kit/world/df8c55dabb35/drift.json`'s
measured drift set, embedded as a literal.

WHY A LITERAL, NOT A RUNTIME READ: `eval/prosecute.py`'s `prosecute()` is
"SYNCHRONOUS, no I/O, no network" (RULES.md section 3's `Gateway.decide`
constraint applies here identically — CONTRACTS.md section 6.1 states the
same for the prosecutor). Reading `kit/world/<id>/drift.json` at claim time
would violate that, and would also make `_hook_stale_read` silently
different depending on which world happens to be mounted locally vs. in a
scored duel. This file is generated OFFLINE, once, from the real exported
world (`kit.world.loader.World.drifts(path_id)` for every path_id
`drift.json` names) — CORPUS-FACTS.md section 2's own measured finding
(~a third of days do not drift at all) is why this is a SUBSET of all
known path_ids, not "every path_id CONTRACTS ever mentions."

Regenerate with:

    python3 -c "
    import json
    drift = json.load(open('kit/world/df8c55dabb35/drift.json'))
    ids = sorted(k for k, v in drift.items() if isinstance(v, dict) and v.get('drifts'))
    print(ids)
    "

`d8f95a7b` (day18) is in this set — the same path_id CORPUS-FACTS.md and
`deck/README.md` both single out (45 working content frames vs. 31
canonical) — and `fixtures/prosecution/labelled/family_a_infrastructure
.jsonl`'s own `stale_read` fixtures are built against it, so this table's
correctness is exercised by `score_prosecutor`, not just asserted.

Stdlib only. No network, no randomness, no wall-clock.
"""

from __future__ import annotations

__all__ = ["WORLD_ID", "DRIFTING_PATH_IDS"]

#: The world this snapshot was taken from — a stale snapshot from a DIFFERENT
#: world_id is still a reasonable prior (better than nothing), but callers
#: that care can compare `exchange_start.p.world_id` against this.
WORLD_ID = "df8c55dabb35"

#: path_id -> drifts (CONTRACTS.md section 2's `drift.json` shape, `Anchor.slug`
#: values only — `Frame:`/`Deck:`/`Section:` namespaces share this key space).
DRIFTING_PATH_IDS: frozenset[str] = frozenset(
    {
        "053195a5",
        "0e11ae43",
        "170c4d17",
        "28e68faa",
        "3326cb76",
        "45098556",
        "72f75709",
        "75811e75",
        "7a8d8046",
        "a13b20e6",
        "a284ae8b",
        "abd20c68",
        "c5418ab5",
        "d8f95a7b",
        "e0614beb",
        "f2696f5f",
    }
)
