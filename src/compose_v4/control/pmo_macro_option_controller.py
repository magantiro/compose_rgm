"""PMO controller that executes declared macro options and protects their bridges.

`MacroOptionController` is a SUBCLASS of the production `PmoPopulationController`. It
adds nothing to that file, so the frozen proposal, credit and memory paths are reached
through `super()` unchanged and the two arms of the matched comparison differ only in
the flags declared here.

The loop it adds
----------------
1. DECLARE. Once, on the first proposal round, it chains structured syntheses from
   archive parents into complete `G0 -> G_bridge -> ... -> G_destination` macros. Every
   leg is EXECUTED and validated by the production executor at declaration time, at zero
   oracle cost, so the controller commits to the complete macro before it asks whether
   any intermediate deserves to stay in the population. A macro that does not exceed the
   measured single-program realization ceiling is refused: it needs no staging.
2. OFFER. Each round, the leg the option is waiting on is materialized from its stored
   construction and injected into the ordinary structured pool, where it is augmented,
   credited and allocated exactly like any other candidate.
3. PROTECT. While a crossing's window is open, the charged intermediate is guaranteed a
   floor of parent mass and its declared continuation is guaranteed an oracle slot.
4. RELEASE. The moment the destination is charged, or the bounded window runs out.

Admissibility is a FILTER, never a shape score
----------------------------------------------
There is no trajectory-shape objective here at all. A leg is admitted only if the
production executor produces it and every intermediate is a committed valid connected
molecule, and `_materialize_stage` RAISES rather than offering a leg that no longer
reproduces its recorded endpoint. Scoring shape without that filter selects beautiful
paths to the wrong molecule -- MEASURED as the gap between interleaving removing the dip
on 9 of 10 transports and preserving the endpoint on only 6 of 10.

Provenance
----------
Every leg carries `provenance["entry_channel"] == MACRO_OPTION_CHANNEL_TAG`, written at
synthesis time, plus the option id, the leg index and whether that crossing was protected
when the leg was synthesized. A downstream basin-entry ablation groups on those.

Accounting
----------
Every state an option touches is charged through the unchanged `ProgramQueryLedger`, in
the ordinary batch, at the ordinary price. Nothing here defers a reservation, caches a
score outside the ledger, or lets an uncertain call become a free one. The registry only
ever OBSERVES endpoints the ledger has already charged.

Arms
----
`enable_macro_options` turns declaration on; `macro_option_protection` turns the
selection guarantee on. They are separate on purpose: the matched comparison needs an
arm that declares and commits to the SAME options and then leaves them to ordinary
score-based selection, which is the thing being tested. With `enable_macro_options`
False nothing in this class runs and the controller is its parent, unchanged --
including its random streams, because option synthesis draws from a dedicated one.
"""

from __future__ import annotations

import json
from time import perf_counter
from typing import Any

import numpy as np

from compose_v4.control.docking_value import identity
from compose_v4.control.dynamic_program_synthesis_v2 import synthesize_structured_program
from compose_v4.control.dynamic_program_synthesis_v21 import STRUCTURED_CHANNEL, _candidate
from compose_v4.control.edit_program import EditProgram
from compose_v4.control.edit_program_graph import (
    compile_program_graph,
    execute_program_graph,
    program_size_profile,
)
from compose_v4.control.pmo_macro_option import (
    DEFAULT_PARENT_MASS_FLOOR,
    DEFAULT_TROUGH_WIDTH,
    MAX_STAGES,
    MEASURED_REALIZATION_CEILING,
    MacroOption,
    MacroOptionRegistry,
    MacroOptionStage,
    option_identity,
    protected_parent_weights,
    reserve_continuation_slots,
)
from compose_v4.control.pmo_population_controller import PmoPopulationController
from compose_v4.rewrite.trace_shard import decode_state

SCHEMA = "pmo_macro_option_controller_v1"

# The provenance tag a downstream ablation groups on. Exported as a constant so a basin
# entry gate imports the exact string instead of transcribing it -- a transcribed tag
# silently stops matching the moment this one moves.
MACRO_OPTION_CHANNEL_TAG = "macro_option_staged_v1"

DEFAULT_SETTINGS: dict[str, Any] = {
    "max_options": 8,
    "max_stages": MAX_STAGES,
    "realization_ceiling": MEASURED_REALIZATION_CEILING,
    # The window IS the chosen ordering's trough width. MEASURED median 1, MAXIMUM 1.
    "trough_width": DEFAULT_TROUGH_WIDTH,
    "parent_mass_floor": DEFAULT_PARENT_MASS_FLOOR,
    "synthesis_attempts_per_origin": 6,
    "declaration_wall_seconds": 120.0,
    # A macro option must PRUNE before it installs. MEASURED, and the reason this is a
    # requirement rather than a preference: `synthesize_structured_program` is additive
    # (atom_insert:atom_delete 2.42:1, mean +1.16 heavy atoms), so an unconstrained chain
    # grows monotonically -- 6 of 6 declared chains did, 20 -> 25 -> 26 -> 31 -> 36 -- and
    # never passes through a state smaller than its own source. The dip this mechanism
    # exists to protect is prune-then-install, so without this the harness declares
    # options that cannot dip and the measurement is VOID by its own instrument check.
    "require_prune_first": True,
    "prune_attempts_per_leg": 12,
}


class MacroOptionProbe(Exception):
    """Raised by the consumption check from inside the production path.

    Deliberately NOT a ValueError / RuntimeError / KeyError / IndexError / TypeError:
    `_generate_channel_pool` catches `ValueError` per attempt and the campaign catches
    broad exceptions per round, so any of those would be swallowed by the very path the
    probe is trying to observe.
    """


class MacroOptionController(PmoPopulationController):
    """PMO population controller with declared macro options and bounded protection."""

    def __init__(
        self,
        *args,
        enable_macro_options: bool = False,
        macro_option_protection: bool = True,
        macro_option_settings: dict[str, Any] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.enable_macro_options = bool(enable_macro_options)
        self.macro_option_protection = bool(macro_option_protection)
        settings = {**DEFAULT_SETTINGS, **(macro_option_settings or {})}
        if not 2 <= int(settings["max_stages"]) <= MAX_STAGES:
            raise ValueError("macro option stage count is outside the declared maximum")
        self.macro_option_settings = settings
        self.option_registry = MacroOptionRegistry(
            parent_mass_floor=float(settings["parent_mass_floor"])
        )
        # A dedicated stream, so declaration never perturbs the production proposal RNGs
        # and the two arms declare a byte-identical option set.
        self.option_rng = np.random.default_rng(
            np.random.SeedSequence([self.config.seed, 977, 0])
        )
        self.options_declared = False
        self.macro_option_diagnostics: dict[str, Any] = {
            "declaration": None,
            "rounds": [],
        }

    # ---- Declaration ----

    def _archive_entry_for(self, endpoint: str) -> dict[str, Any] | None:
        for _, entry in sorted(self.entries.items()):
            if entry["endpoint"] == endpoint:
                return entry
        return None

    def _measured_score(self, endpoint: str) -> float | None:
        scores = [
            float(row["score"])
            for row in self.observations.values()
            if row["endpoint"] == endpoint
        ]
        return float(np.mean(scores)) if scores else None

    def _synthesize_chain(self, origin_entry: dict[str, Any]) -> dict[str, Any] | None:
        """Chain structured syntheses until the macro exceeds the realization ceiling.

        Returns None when no chain within `max_stages` clears the ceiling. Every leg is
        executed here, so a returned chain is certified by the executor rather than
        planned: this is the commitment the protection is granted against.
        """
        ceiling = int(self.macro_option_settings["realization_ceiling"])
        max_stages = int(self.macro_option_settings["max_stages"])
        prune_first = bool(self.macro_option_settings.get("require_prune_first", True))
        prune_attempts = int(self.macro_option_settings.get("prune_attempts_per_leg", 12))
        current = decode_state(origin_entry["trace"]["states"][-1])
        origin_atoms = current.n_real_atoms
        origin_endpoint = origin_entry["endpoint"]
        stages: list[MacroOptionStage] = []
        construction: list[dict[str, Any]] = []
        parent_endpoint, total = origin_endpoint, 0
        seen = {origin_endpoint}
        heavy_chain = [origin_atoms]
        for index in range(max_stages):
            # The first leg must EXCISE. Read from the executed state, not a SMILES
            # re-parse, so the count is the one the proposal path itself carries.
            attempts = prune_attempts if (prune_first and index == 0) else 1
            leg = None
            for _ in range(attempts):
                try:
                    source, program, binding, _, _ = synthesize_structured_program(
                        current,
                        self.option_rng,
                        max_modules=3,
                        max_primitives=self.config.max_primitives,
                        max_blocks=self.config.max_blocks,
                        panel_cache=self._v21_panel_cache,
                    )
                    graph = compile_program_graph(program)
                    _, trace = execute_program_graph(
                        source,
                        graph,
                        binding,
                        max_primitives=self.config.max_primitives,
                        max_blocks=self.config.max_blocks,
                    )
                except (ValueError, RuntimeError):
                    continue
                produced = decode_state(trace["states"][-1]).n_real_atoms
                if prune_first and index == 0 and produced >= origin_atoms:
                    continue
                leg = (source, program, binding, trace, produced)
                break
            if leg is None:
                return None
            source, program, binding, trace, produced = leg
            endpoint = trace["endpoint"]
            primitives = len(trace["actions"])
            if not primitives or endpoint in seen:
                # A leg that does no work, or that returns to a state the chain has
                # already passed through, is not a transport.
                return None
            seen.add(endpoint)
            heavy_chain.append(produced)
            stages.append(MacroOptionStage(index, parent_endpoint, endpoint, primitives))
            construction.append(
                {
                    "source_state": trace["states"][0],
                    "program": program.payload(),
                    "assignment": list(binding),
                    "endpoint": endpoint,
                    "primitives": primitives,
                }
            )
            total += primitives
            parent_endpoint = endpoint
            current = decode_state(trace["states"][-1])
            if total > ceiling and len(stages) >= 2:
                return {
                    "origin_endpoint": origin_endpoint,
                    "stages": stages,
                    "construction": {"stages": construction},
                    "total_primitives": total,
                    # Recorded so a reader can check the option really prunes before it
                    # installs, instead of trusting the flag that asked for it.
                    "heavy_atom_chain": heavy_chain,
                    "prunes_before_installing": heavy_chain[1] < heavy_chain[0],
                }
        return None

    def _declare_macro_options(self) -> dict[str, Any]:
        """Declare up to `max_options` complete validated macros. Zero oracle calls."""
        began = perf_counter()
        wall = float(self.macro_option_settings["declaration_wall_seconds"])
        attempts_per_origin = int(self.macro_option_settings["synthesis_attempts_per_origin"])
        limit = int(self.macro_option_settings["max_options"])
        protection = int(self.macro_option_settings["trough_width"])
        ceiling = int(self.macro_option_settings["realization_ceiling"])
        origins = [entry for _, entry in sorted(self.entries.items())]
        prune_first = bool(self.macro_option_settings.get("require_prune_first", True))
        declared, refused = [], {"no_chain_over_ceiling": 0, "duplicate_identity": 0}
        heavy_chains: list[list[int]] = []
        for origin in origins:
            if len(declared) >= limit or perf_counter() - began >= wall:
                break
            for _ in range(attempts_per_origin):
                if len(declared) >= limit or perf_counter() - began >= wall:
                    break
                chain = self._synthesize_chain(origin)
                if chain is None:
                    refused["no_chain_over_ceiling"] += 1
                    continue
                option = MacroOption(
                    option_id=option_identity(
                        chain["origin_endpoint"],
                        [stage.endpoint for stage in chain["stages"]],
                    ),
                    origin_endpoint=chain["origin_endpoint"],
                    stages=tuple(chain["stages"]),
                    protection_rounds=protection,
                    declared_at_round=int(self.batches),
                    realization_ceiling=ceiling,
                )
                if option.option_id in self.option_registry.options:
                    refused["duplicate_identity"] += 1
                    continue
                self.option_registry.declare(option, chain["construction"])
                declared.append(
                    {
                        **option.payload(),
                        "heavy_atom_chain": chain["heavy_atom_chain"],
                        "prunes_before_installing": chain["prunes_before_installing"],
                    }
                )
                heavy_chains.append(chain["heavy_atom_chain"])
                break
        self.options_declared = True
        report = {
            "schema_version": SCHEMA,
            "declared_at_round": int(self.batches),
            "origins_available": len(origins),
            "options": declared,
            "refusals": refused,
            "seconds": perf_counter() - began,
            "oracle_calls": 0,
            "realization_ceiling": ceiling,
            "trough_width_rounds": protection,
            "protection_source": "chosen_ordering_trough_width",
            "channel_tag": MACRO_OPTION_CHANNEL_TAG,
            "require_prune_first": prune_first,
            "heavy_atom_chains": heavy_chains,
            "options_that_prune_before_installing": sum(
                1 for row in declared if row["prunes_before_installing"]
            ),
        }
        self.macro_option_diagnostics["declaration"] = report
        return report

    # ---- Offering declared legs into the ordinary pool ----

    def _materialize_stage(
        self, option_id: str, stage_index: int, eligibility
    ) -> dict[str, Any] | None:
        option = self.option_registry.options[option_id]
        stored = self.option_registry.construction(option_id)["stages"][stage_index]
        parent_endpoint = option.stages[stage_index].parent_endpoint
        entry = self._archive_entry_for(parent_endpoint)
        measured = self._measured_score(parent_endpoint)
        if entry is None or measured is None:
            # The leg's parent is not yet a scored member of the population, so there is
            # nothing to continue from. Not an error: the option simply waits.
            return None
        source = decode_state(stored["source_state"])
        program = EditProgram.from_payload(stored["program"])
        binding = tuple(stored["assignment"])
        graph = compile_program_graph(program)
        try:
            _, trace = execute_program_graph(
                source,
                graph,
                binding,
                max_primitives=self.config.max_primitives,
                max_blocks=self.config.max_blocks,
            )
        except (ValueError, RuntimeError):
            return None
        if trace["endpoint"] != stored["endpoint"]:
            # The executor is the authority. A stored leg that no longer reproduces its
            # recorded endpoint is not offered, rather than quietly offering a different
            # molecule under a declared option's name.
            raise ValueError("declared macro option leg no longer reproduces its endpoint")
        size = program_size_profile(graph, source.n_real_atoms)
        measured_atoms = decode_state(entry["trace"]["states"][-1]).n_real_atoms
        size["measured_parent_heavy_atoms"] = measured_atoms
        size["delta_from_measured_parent"] = size["final_heavy_atoms"] - measured_atoms
        properties = eligibility({"smiles": trace["endpoint"]})
        if type(properties.get("oracle_eligible")) is not bool:
            raise ValueError("endpoint evaluator must return explicit boolean eligibility")
        if not properties["oracle_eligible"]:
            return None
        parent = {
            "entry_id": entry["entry_id"],
            "parent_probability": 0.0,
            "parent_measured_score": measured,
        }
        candidate = _candidate(
            optimizer=self,
            entry=entry,
            parent=parent,
            channel=STRUCTURED_CHANNEL,
            attempt=-1,
            source=source,
            program=program,
            binding=binding,
            trace=trace,
            metadata={"macro_option_leg": {"option_id": option_id, "stage": stage_index}},
            size=size,
            properties=properties,
        )
        # Provenance written HERE, at synthesis time, on the candidate record itself, so
        # a downstream ablation can attribute a productive-region entrant to this channel
        # and separate it from donor transport and from broad exploration. It rides on the
        # record `add_measured_program` archives, so it survives into the snapshot.
        candidate["provenance"]["entry_channel"] = MACRO_OPTION_CHANNEL_TAG
        candidate["provenance"]["macro_option_id"] = option_id
        candidate["provenance"]["macro_option_stage"] = stage_index
        candidate["provenance"]["macro_option_stages_total"] = len(option.stages)
        candidate["provenance"]["macro_option_is_destination"] = (
            stage_index == len(option.stages) - 1
        )
        # Whether the crossing this leg continues was PROTECTED when it was synthesized.
        # The ablation needs the mechanism that actually applied, not the arm's label.
        candidate["provenance"]["macro_option_protected"] = bool(
            self.macro_option_protection
            and option_id in set(self.option_registry.protected_bridges().values())
        )
        candidate["provenance"]["macro_option_total_primitives"] = option.total_primitives
        return candidate

    def _macro_option_candidates(
        self, eligibility, archive_seen, drawn_parents: set[str]
    ) -> list[dict[str, Any]]:
        """Offer the leg each option is waiting on -- but ONLY from a drawn parent.

        Gating on the round's own parent schedule is what makes the parent-mass floor
        load-bearing.  Offering a stored leg unconditionally would let the option advance
        from a bridge the selection law never drew, which is precisely the starvation
        being repaired: the continuation would arrive for free and the floor would be
        decoration.  In production a continuation exists only because a parent was
        chosen, and the same must hold here.
        """
        rows = []
        for option_id in sorted(self.option_registry.options):
            if self.option_registry.status(option_id) in ("reached", "expired"):
                continue
            stage_index = self.option_registry.next_stage_index(option_id)
            option = self.option_registry.options[option_id]
            if stage_index >= len(option.stages):
                continue
            stage = option.stages[stage_index]
            if stage.endpoint in archive_seen or stage.parent_endpoint not in drawn_parents:
                continue
            candidate = self._materialize_stage(option_id, stage_index, eligibility)
            if candidate is not None:
                rows.append(candidate)
        return rows

    def _generate_channel_pool(self, channel, eligibility, archive_seen, parent_schedule):
        attempts, candidates, elapsed = super()._generate_channel_pool(
            channel, eligibility, archive_seen, parent_schedule
        )
        if not self.enable_macro_options or channel != STRUCTURED_CHANNEL:
            return attempts, candidates, elapsed
        began = perf_counter()
        seen = set(archive_seen) | {row["endpoint"] for row in candidates}
        drawn = {entry["endpoint"] for entry, _ in parent_schedule}
        self.macro_option_diagnostics["last_drawn_parents"] = len(drawn)
        for candidate in self._macro_option_candidates(eligibility, seen, drawn):
            if candidate["endpoint"] in seen:
                continue
            seen.add(candidate["endpoint"])
            candidates.append(candidate)
            attempts.append(
                {
                    "attempt": -1,
                    "planner_channel": channel,
                    "status": "eligible",
                    "endpoint": candidate["endpoint"],
                    "macro_option_id": candidate["provenance"]["macro_option_id"],
                    "macro_option_stage": candidate["provenance"]["macro_option_stage"],
                }
            )
        return attempts, candidates, elapsed + (perf_counter() - began)

    # ---- Selection: the parent-mass floor ----

    def selection(self):
        keys, weights = super().selection()
        if not (self.enable_macro_options and self.macro_option_protection):
            return keys, weights
        protected = self.option_registry.protected_bridges()
        if not protected:
            return keys, weights
        endpoint_of = {key: self.entries[key]["endpoint"] for key in keys}
        lifted, _ = protected_parent_weights(
            keys,
            weights,
            endpoint_of,
            protected,
            floor=self.option_registry.parent_mass_floor,
        )
        return keys, lifted

    # ---- Allocation: the reserved continuation slot ----

    def _reserved_candidate_ids(self, candidates: list[dict[str, Any]]) -> set[str]:
        """Which option legs this round must charge.

        Stage 0 is reserved in BOTH arms: charging the first intermediate is the
        COMMITMENT to the macro, and the matched comparison isolates protection, not
        commitment. Every later leg is reserved only while its crossing's window is
        open, which is the mechanism under test.
        """
        protected_options = set(self.option_registry.protected_bridges().values())
        reserved = set()
        for row in candidates:
            option_id = row["provenance"].get("macro_option_id")
            if option_id is None:
                continue
            stage = int(row["provenance"]["macro_option_stage"])
            commitment = stage == 0
            continuation = self.macro_option_protection and option_id in protected_options
            if commitment or continuation:
                reserved.add(row["candidate_id"])
        return reserved

    def _allocate(self, candidates: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict]:
        chosen, detail = super()._allocate(candidates)
        if not self.enable_macro_options:
            return chosen, detail
        reserved = self._reserved_candidate_ids(candidates)
        limit = min(self.config.candidates_per_batch, len(candidates))
        merged, reservation = reserve_continuation_slots(
            chosen, candidates, reserved, limit=limit
        )
        detail = {
            **detail,
            "macro_option_reservation": {
                **reservation,
                "protection_enabled": self.macro_option_protection,
                "selected_ids": [row["candidate_id"] for row in merged],
            },
        }
        detail["selected_ids"] = [row["candidate_id"] for row in merged]
        detail["selection_role_by_candidate"] = {
            row["candidate_id"]: (
                "macro_option_reserved"
                if row["candidate_id"] in reserved
                else detail["selection_role_by_candidate"].get(row["candidate_id"], "displaced")
            )
            for row in merged
        }
        return merged, detail

    # ---- Round clock and charge observation ----

    def _reconcile_with_archive(self, round_index: int) -> None:
        """Advance any option whose next leg the ordinary search already charged."""
        observed = {row["endpoint"] for row in self.observations.values()}
        for option_id in sorted(self.option_registry.options):
            if self.option_registry.status(option_id) in ("reached", "expired"):
                continue
            option = self.option_registry.options[option_id]
            index = self.option_registry.next_stage_index(option_id)
            while index < len(option.stages) and option.stages[index].endpoint in observed:
                self.option_registry.note_charged(option.stages[index].endpoint, round_index)
                index = self.option_registry.next_stage_index(option_id)

    def propose_batch(self, eligibility):
        if self.enable_macro_options:
            if not self.options_declared:
                self._declare_macro_options()
            self._reconcile_with_archive(int(self.batches))
            self.option_registry.advance_round(int(self.batches))
        batch = super().propose_batch(eligibility)
        if self.enable_macro_options:
            self.macro_option_diagnostics["rounds"].append(
                {
                    "round": int(self.batches),
                    "protected": sorted(self.option_registry.protected_bridges().values()),
                    "status_counts": self.option_registry.report()["status_counts"],
                }
            )
        return batch

    def observe_batch(self, batch_id, outcomes):
        charged = []
        if self.enable_macro_options and self.pending is not None:
            candidates = {row["candidate_id"]: row for row in self.pending["candidates"]}
            charged = [
                candidates[row["candidate_id"]]["endpoint"]
                for row in outcomes
                if row.get("candidate_id") in candidates and row.get("score") is not None
            ]
        round_index = int(self.batches)
        super().observe_batch(batch_id, outcomes)
        for endpoint in charged:
            # Observation only. The ledger already charged these; the registry never
            # performs, defers or forgives a query.
            self.option_registry.note_charged(endpoint, round_index)

    # ---- Reporting and durable state ----

    def macro_option_report(self) -> dict[str, Any]:
        report = self.option_registry.report()
        return {
            **report,
            "schema_version": SCHEMA,
            "enabled": self.enable_macro_options,
            "protection_enabled": self.macro_option_protection,
            "settings": dict(self.macro_option_settings),
            "declaration": self.macro_option_diagnostics["declaration"],
        }

    def snapshot(self, *, include_history=True):
        snapshot = super().snapshot(include_history=include_history)
        body = {key: value for key, value in snapshot.items() if key != "snapshot_id"}
        body["macro_options"] = {
            "schema_version": SCHEMA,
            "enabled": self.enable_macro_options,
            "protection_enabled": self.macro_option_protection,
            "settings": dict(self.macro_option_settings),
            "options_declared": self.options_declared,
            "option_rng": self.option_rng.bit_generator.state,
            # Without the registry a preempted run resumes with no open windows and
            # every in-flight option is stranded: its bridge is already charged and
            # nothing would ever continue from it.
            "registry": self.option_registry.payload(),
            "diagnostics": self.macro_option_diagnostics,
        }
        return json.loads(json.dumps({**body, "snapshot_id": identity(body)}))

    @classmethod
    def restore(
        cls,
        snapshot,
        *,
        hierarchy=None,
        jump_checkpoint=None,
        enable_online_memory: bool = False,
        enable_macro_options: bool = False,
        macro_option_protection: bool = True,
        macro_option_settings: dict[str, Any] | None = None,
    ):
        # `run_program_campaign` passes optimizer_kwargs to BOTH the constructor and this
        # classmethod, so every arm flag has to be accepted here too. A flag silently
        # dropped on the resume path rebuilds the wrong arm and the artifact cannot tell.
        result = super().restore(
            snapshot,
            hierarchy=hierarchy,
            jump_checkpoint=jump_checkpoint,
            enable_online_memory=enable_online_memory,
        )
        result.enable_macro_options = bool(enable_macro_options)
        result.macro_option_protection = bool(macro_option_protection)
        if macro_option_settings:
            result.macro_option_settings = {**DEFAULT_SETTINGS, **macro_option_settings}
        state = snapshot.get("macro_options")
        if state is None:
            if enable_macro_options:
                raise ValueError(
                    "snapshot carries no macro-option state but this controller was "
                    "constructed with options enabled; resuming would silently change the arm"
                )
            return result
        if bool(state["enabled"]) != result.enable_macro_options or bool(
            state["protection_enabled"]
        ) != result.macro_option_protection:
            # Resuming a protected run as an unprotected one, or the reverse, is the
            # silent arm change the whole matched comparison exists to prevent.
            raise ValueError("macro-option arm flags changed across resume")
        result.macro_option_settings = {**DEFAULT_SETTINGS, **state["settings"]}
        result.options_declared = bool(state["options_declared"])
        result.option_rng.bit_generator.state = state["option_rng"]
        result.option_registry = MacroOptionRegistry.restore(state["registry"])
        result.macro_option_diagnostics = json.loads(json.dumps(state["diagnostics"]))
        return result


def assert_macro_option_protection_is_consumed(run) -> dict[str, Any]:
    """Prove the PRODUCTION path reaches both protection hooks. Raises if it does not.

    A validated mechanism behind a flag is INERT until a caller passes it, and "the
    module has tests" hides that completely -- this repository has paid for that three
    times. `inspect.signature` would not catch it either: in every one of those cases the
    parameter existed and was dropped one hop later.

    So this RUNS the caller's own closure with a probe installed on the module globals
    the controller resolves at call time, and requires the production path to reach it.
    The probe raises `MacroOptionProbe`, which is deliberately outside the exception
    types `_generate_channel_pool` and the campaign catch per attempt and per round: a
    `ValueError` would be swallowed by the very path being observed.
    """
    module = globals()
    reached: dict[str, bool] = {"parent_mass_floor": False, "reserved_slot": False}
    refused: dict[str, str] = {}

    def probe_floor(*args, **kwargs):
        reached["parent_mass_floor"] = True
        raise MacroOptionProbe("parent-mass floor reached")

    def probe_reserve(*args, **kwargs):
        reached["reserved_slot"] = True
        raise MacroOptionProbe("continuation reservation reached")

    original = module["protected_parent_weights"], module["reserve_continuation_slots"]
    for name, probe in (
        ("protected_parent_weights", probe_floor),
        ("reserve_continuation_slots", probe_reserve),
    ):
        module[name] = probe
        try:
            run()
        except MacroOptionProbe:
            pass
        except Exception as error:  # noqa: BLE001
            # Recorded, never swallowed silently, and it can only push the verdict
            # TOWARDS "not consumed": a flag is set only by the probe being entered, so
            # an unraised closure cannot manufacture a pass. `run` is called once per
            # probe, so a closure that mutates shared state across calls -- a controller
            # whose `propose_batch` completes and leaves a pending batch, say -- must
            # build it fresh. That is why this is reported rather than raised: the
            # diagnosis belongs in the report, not in a traceback from the wrong layer.
            refused[name] = repr(error)
        finally:
            module["protected_parent_weights"], module["reserve_continuation_slots"] = original
    if not all(reached.values()):
        unreached = sorted(key for key, value in reached.items() if not value)
        raise ValueError(
            "macro option protection is NOT consumed by the production path: "
            f"{unreached}"
            + (f" (closure refused: {refused})" if refused else "")
        )
    return {"consumed": reached, "closure_refusals": refused, "schema_version": SCHEMA}


__all__ = [
    "DEFAULT_SETTINGS",
    "MACRO_OPTION_CHANNEL_TAG",
    "SCHEMA",
    "MacroOptionController",
    "MacroOptionProbe",
    "assert_macro_option_protection_is_consumed",
]
