"""`mea` command line: plan, run, snake, conditions, compile, rawqc, import.

Recordings are discovered by suffix chain (`_raw`, `_cleaned`, `_QCmanual`, ...)
and enter at their furthest state-bearing file; `--stage X` rewinds upstream of X.
"""
from __future__ import annotations
import argparse, dataclasses, json, subprocess, sys, textwrap, time

import numpy as np
from pathlib import Path

from . import config as cfgmod
from .discovery import scan
from .pipeline import BY_KEY, ORDER, import_entry, import_source
from .io.importer import import_mat, can_run

GREEN, GREY, RED, YELLOW, OFF = "\033[32m", "\033[90m", "\033[31m", "\033[33m", "\033[0m"


def _plan_for(rec, cfg):
    """Rows of (stage, verdict, reason) for one discovered recording."""
    rows = []
    have = set()
    for s in ORDER:
        if s in rec.outputs:
            have.add(BY_KEY[s].produces)
    # sorter output grants what stage 1 grants
    if rec.sorter_output is not None:
        have |= {"spikes", "conditions", "pre_echo_spikes"}
    # what the furthest existing output can grant
    if rec.outputs:
        have |= {"spikes", "conditions"}
        if "qc_general" in rec.outputs:
            have |= {"qc.general", "pre_echo_spikes"}
        if "qc_manual" in rec.outputs or "integrate" in rec.outputs:
            have |= {"qc.manual"}
    # stages at or before the furthest one count as done (QC outputs are often deleted)
    furthest_i = ORDER.index(rec.furthest) if rec.furthest else -1
    for key in ORDER:
        st = BY_KEY[key]
        i = ORDER.index(key)
        if key in rec.outputs:
            rows.append((key, "have", "output already present"))
            continue
        if key == "shape" and rec.sorter_output is not None:
            rows.append((key, "have", f"sorter output in {rec.sorter_output.name}/"))
            continue
        if i < furthest_i:
            rows.append((key, "have", "superseded by a later output"))
            continue
        if not st.implemented:
            rows.append((key, "pending", st.note or "not implemented yet"))
            continue
        missing = st.requires - have
        if missing:
            rows.append((key, "blocked", "missing " + ", ".join(sorted(missing))))
        else:
            rows.append((key, "run", st.note))
    return rows


def cmd_plan(args):
    cfg = cfgmod.load(args.config, args.set)
    found = scan(args.input, cfg)
    if not found:
        print("no recordings found"); return 1
    for name, rec in found.items():
        print(f"\n{name}   [{rec.directory}]")
        for key, verdict, reason in _plan_for(rec, cfg):
            col = {"have": GREY, "run": GREEN, "blocked": RED, "pending": YELLOW}[verdict]
            print(f"   {col}{verdict:8s}{OFF} {key:14s} {GREY}{reason}{OFF}")
    return 0


def cmd_run(args):
    cfg = cfgmod.load(args.config, args.set)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    found = scan(args.input, cfg)
    rc = 0
    quantified = 0
    for name, rec in found.items():
        print(f"\n=== {name} ===")
        plan = {k: (v, r) for k, v, r in _plan_for(rec, cfg)}
        # Everything from the first runnable stage on; each is re-checked live below
        if args.stage:
            only = [args.stage]
        else:
            runnable = [i for i, k in enumerate(ORDER) if plan[k][0] == "run"]
            only = ORDER[runnable[0]:] if runnable else []
        # compile runs once after the loop
        only = [k for k in only if k != "compile"]
        if not only:
            print("   nothing to run"); continue
        # Enter from the furthest state-bearing output (_Analyzed/_Compiled lack conditions)
        stage_in, src = import_source(rec, before=args.stage)
        if src is None:
            hint = f" upstream of {args.stage}" if args.stage else ""
            print(f"   {RED}no usable input{OFF}{hint}"); rc = 1; continue
        r = import_entry(src)
        r.name = name                     # the discovery name, not the file's
        print(f"   imported {src.name}  ({r.n_units} units, {r.t_ms.size:,} spikes)"
              f"  [entered at {stage_in}]")
        ctx = r.context()
        stage_results = []
        considered = None                 # the recording the QC stages judged
        for key in only:
            st = BY_KEY[key]
            if not st.implemented:
                print(f"   {YELLOW}skip{OFF} {key}: {st.note}"); continue
            ok, missing = can_run(r, key) if key in ("qc_general", "qc_crosstalk",
                                                     "qc_manual", "quantify") else (True, set())
            if not ok:
                print(f"   {RED}skip{OFF} {key}: missing {', '.join(sorted(missing))}"); continue
            t0 = time.perf_counter()
            try:
                res = st.run(r, cfg, ctx)
            except Exception as e:
                print(f"   {RED}fail{OFF} {key}: {type(e).__name__}: {e}"); rc = 1; continue
            ctx[key] = res

            # Drop units this stage marked discarded before the next stage runs
            if key in ("qc_general", "qc_manual", "integrate"):
                # keep the pre-filter recording for diagnostic figures
                if considered is None:
                    considered = r
                before = r.n_units
                mask_before = r.kept_mask()
                r = r.subset(mask_before)
                if r.n_units != before:
                    print(f"        {before} -> {r.n_units} units")
                # Rebuild per-unit ctx for the smaller recording
                counts, lengths = r.per_condition()
                ctx["per_condition_counts"] = counts
                ctx["per_condition_length_s"] = lengths
                if key == "qc_general":
                    # pre-echo-removal trains of the stage-2 survivors, for stage 3
                    kept = np.flatnonzero(mask_before)
                    trains = res.get("trains")
                    ctx["pre_echo_trains"] = ([trains[i] for i in kept]
                                              if trains is not None else
                                              [r.train(i) for i in range(r.n_units)])

            # Grant the stage's capability ("qc.general", not the stage name) to unblock the next
            caps = set(r.meta.get("capabilities", [])) | {st.produces}
            if key == "qc_general":
                # pre_echo_trains already set above from survivors; don't use res["trains"]
                caps.add("pre_echo_spikes")
            if key == "qc_crosstalk" and "discard_mask" in res:
                ctx["crosstalk_scores"] = res["discard_mask"]
            r.meta["capabilities"] = sorted(caps)
            if isinstance(res, dict):
                stage_results.append((key, res))
            print(f"   {GREEN}ok{OFF}   {key:14s} {time.perf_counter()-t0:6.2f}s")
        d = r.save(out)
        r.meta["_dir"] = str(d)
        for key, res in stage_results:
            r.persist(key, res)
        r.save(out)                       # meta now carries capabilities/results
        print(f"   wrote {d}")

        # Figures last, after the store is written
        if cfg.get("plots", {}).get("enabled", True) and not args.no_plots:
            from .plots import write_all
            # Mark the final verdict on all judged units so figures show discarded vs kept
            seen = considered
            if seen is not None and seen is not r:
                survivors = set(r.units["unit_id"])
                u = seen.units.copy()
                u["kept"] = u["unit_id"].isin(survivors)
                # survivors of the non-TTX tests, for ttx_control
                qcres = ctx.get("qc_general")
                if qcres and "unitType_bombcell" in u.columns:
                    from .stages.s02_qc_general import ALL_TESTS, non_ttx_survivors
                    g = cfg.get("qc_general", {})
                    keep_types = cfg.get("integrate", {}).get(
                        "unit_select_kilosort", [1, 2, 3, 4])
                    sel = u["unitType_bombcell"].isin(keep_types).to_numpy()
                    u["kept_non_ttx"] = non_ttx_survivors(
                        qcres, sel, "ttx_sensitive" in qcres,
                        tuple(g.get("tests", ALL_TESTS)))
                seen = dataclasses.replace(seen, units=u)
            pr = write_all(r, cfg, ctx.get("quantify"), out_dir=d,
                           candidates=seen)
            if pr.get("written"):
                print(f"   {len(pr['written'])} figures -> {pr['dir']}")
            for name, why in (pr.get("failed") or {}).items():
                print(f"   {YELLOW}figure {name} failed{OFF}: {why}")
        quantified += 1

    # --- compile across recordings ---
    if quantified and not args.stage:
        print(f"\n=== compile ===")
        from .stages import s07_compile as comp
        try:
            res = comp.run(out, cfg, Path(args.compile_out or out / "compiled"))
        except Exception as e:                                   # noqa: BLE001
            print(f"   {RED}fail{OFF} compile: {type(e).__name__}: {e}")
            return 1
        if res["n_recordings"]:
            print(f"   {res['n_recordings']} recordings, {res['n_conditions']} "
                  f"conditions, {res['n_metrics']} metrics")
            _report_thin(res)
            for w in res["written"]:
                print(f"   wrote {w}")
    return rc


def cmd_conditions(args):
    """List each recording's condition files with the index used by dataset.discovery.order.

    Shows acquisition time and mtime; mtime belongs to the copy, not the experiment.
    """
    from .conditions import describe
    from . import conditions as C
    cfg = cfgmod.load(args.config, args.set)
    found = scan(args.input, cfg)
    shown = 0
    for name, rec in found.items():
        for kind, paths in rec.raw.items():
            if not paths:
                continue
            rows, mode = describe(paths, cfg)
            shown += 1
            print(f"\n{name}   [{kind}]   base order: {mode}")
            print(f"   {'#':>3}  {'file':<52} {'acquired':<20} {'modified':<20}")
            for r in rows:
                print(f"   {r['index']:>3}  {r['name']:<52} {r['acquired']:<20} {r['modified']:<20}")
            try:
                ordered, how = C.order_condition_files(paths, cfg)
                print(f"   -> conditions in order ({how}):")
                for i, p in enumerate(ordered, 1):
                    print(f"        {i}. {p.name}")
            except ValueError as e:
                print(f"   {RED}order is not usable{OFF}: {e}")
                return 1
    if not shown:
        print("no raw condition files found; "
              "condition order comes from rec_conditions inside a .mat")
    return 0


def _report_thin(res):
    """Conditions whose network values rest on little or nothing."""
    why = []
    if res.get("n_not_bursting"):
        why.append(f"{res['n_not_bursting']} not network bursting")
    if res.get("n_few_events"):
        why.append(f"{res['n_few_events']} with fewer than 3 events")
    if why:
        print(f"   {YELLOW}conditions with thin network values{OFF}: "
              f"{', '.join(why)} (see network.nBurst)")


def cmd_compile(args):
    """Join every recording's metrics under a results root into one table."""
    from .stages import s07_compile as m
    cfg = cfgmod.load(args.config, args.set)
    res = m.run(args.input, cfg, args.out, sample_sheet=args.samples)
    if not res["n_recordings"]:
        print(f"   {YELLOW}nothing to compile{OFF}: no quantified recordings under {args.input}")
        return 1
    print(f"   {res['n_recordings']} recordings, {res['n_conditions']} conditions, "
          f"{res['n_metrics']} metrics")
    _report_thin(res)
    if res.get("not_in_sheet"):
        print(f"   {YELLOW}not in --samples{OFF}: {', '.join(res['not_in_sheet'])}")
    for w in res["written"]:
        print(f"   wrote {w}")
    return 0


def cmd_rawqc(args):
    """Pre-sorting chip screen: pass if TTX silences enough electrodes (raw headers only)."""
    from .rawqc import ttx_gate
    from .dataset import label_conditions, role_mask
    from . import conditions as C

    cfg = cfgmod.load(args.config, args.set)
    found = scan(args.input, cfg)
    rows, failures, skipped = [], 0, 0
    for name, rec in sorted(found.items()):
        paths = rec.raw.get("shape_kilosort") or []
        if len(paths) < 2:
            continue
        ordered, _ = C.order_condition_files(paths, cfg)
        labels = label_conditions([p.name for p in ordered], cfg)
        ttx = role_mask(labels, cfg, "silencing")
        base = role_mask(labels, cfg, "baseline")
        if ttx.sum() != 1 or base.sum() != 1:
            print(f"{name:34s} {YELLOW}skipped{OFF}  needs exactly one baseline and "
                  f"one TTX condition; found {int(base.sum())} and {int(ttx.sum())}")
            skipped += 1
            continue
        # own untreated baseline only: a drug condition would inflate the active set
        q = ttx_gate(ordered[int(base.argmax())], ordered[int(ttx.argmax())], cfg)
        rows.append((name, q))
        failures += not q.passed

    if not rows and not skipped:
        print("no recordings with both a baseline and a TTX condition found")
        return 0
    print(f"\n   {'recording':<34} {'active':>7} {'TTX-sensitive':>14} {'':>6}")
    for name, q in rows:
        mark = f"{GREEN}pass{OFF}" if q.passed else f"{RED}fail{OFF}"
        print(f"   {name:<34} {q.n_active:>7} {q.n_ttx_sensitive:>14}   {mark}")
    print(f"\n   {len(rows) - failures} of {len(rows)} pass at "
          f"min_ttx_sensitive = {rows[0][1].threshold if rows else '-'}")
    return 1 if (failures and args.strict) else 0


def cmd_snake(args):
    """Hand the same work to Snakemake: parallel across recordings, resumable."""
    sf = Path(__file__).resolve().parents[2] / "workflow" / "Snakefile"
    cfg = [f"input={json.dumps(list(args.input))}", f"out={args.out}"]
    if args.config:
        cfg.append(f"config={args.config}")
    cmd = ["snakemake", "-s", str(sf), "-j", str(args.jobs), "--config", *cfg]
    if args.dry_run:
        cmd.append("--dry-run")
    print("  " + " ".join(cmd))
    return subprocess.call(cmd)


def cmd_import(args):
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    try:
        r = import_mat(args.file)
    except ValueError as e:
        print(f"{RED}cannot import{OFF} {Path(args.file).name}: {e}")
        return 1
    d = r.save(out)
    print(f"{args.file} -> {d}")
    print(f"   {r.n_units} units, {r.t_ms.size:,} spikes, {len(r.conditions)} conditions")
    print(f"   this file is the output of stage: {r.meta['produced_by']}")
    print(f"   capabilities: {', '.join(r.meta['capabilities'])}")
    return 0


def main(argv=None):
    fmt = argparse.RawDescriptionHelpFormatter
    p = argparse.ArgumentParser(
        prog="mea", formatter_class=fmt,
        description="MEA spike-train analysis pipeline "
                    "(Python port of the Yang MATLAB pipeline).",
        epilog=textwrap.dedent("""\
            typical session:
              mea conditions --input DATA          check which file is which condition
              mea plan       --input DATA          see what would run, and what is blocked
              mea run        --input DATA --out R  run it
              mea snake      --input DATA --out R -j 8   the same, in parallel

            You point mea at DIRECTORIES, not files.  Recordings are found by their
            filename stem and each one enters the pipeline where its MATLAB run left
            off.  Results go to R/<recording>.mea/ as parquet + json.

            Every parameter lives in config/default.yaml.  Override one with --set,
            or a whole file with --config.
            """))
    p.add_argument("--config", metavar="FILE",
                   help="YAML applied on top of config/default.yaml")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="override one config value; repeatable. "
                        "e.g. --set network_burst.threshold_sd=4.0")
    sub = p.add_subparsers(dest="cmd", required=True, metavar="COMMAND")

    # ---------------------------------------------------------------- plan
    q = sub.add_parser(
        "plan", formatter_class=fmt, help="show what would run (no changes made)",
        description="Report, per recording, which stages are already done, which "
                    "would run, and which are blocked and why.  Makes no changes.",
        epilog=textwrap.dedent("""\
            status column:
              have     the MATLAB already produced this stage's output
              run      would run now
              blocked  a required input is missing (the reason is printed)
              pending  the stage is not ported yet

            examples:
              mea plan --input /data/batch4
              mea plan --input /data/batch4 /data/batch5
            """))
    q.add_argument("--input", nargs="+", required=True, metavar="DIR",
                   help="one or more directories to scan for recordings")
    q.set_defaults(fn=cmd_plan)

    # ----------------------------------------------------------------- run
    q = sub.add_parser(
        "run", formatter_class=fmt, help="run the pipeline in this process",
        description="Run every stage that is ready, one recording at a time, "
                    "writing a .mea store per recording.",
        epilog=textwrap.dedent("""\
            Each recording enters at the point its MATLAB run left off.  --stage
            rewinds further: it enters from the last file UPSTREAM of that stage and
            runs only it.

              --stage qc_general   enters at *_raw.mat        (all channels, pre-QC)
              --stage qc_manual    enters at *_cleaned.mat    (post-QC, uncurated)
              --stage quantify     enters at *_curated_QCmanual.mat

            examples:
              mea run --input /data/batch4 --out results
              mea run --input /data/batch4 --out results --stage qc_general
              mea run --input /data/batch4 --out results --set experiment.window_s=300

            For more than a couple of recordings prefer `mea snake`, which runs them
            in parallel and can resume.
            """))
    q.add_argument("--input", nargs="+", required=True, metavar="DIR",
                   help="one or more directories to scan for recordings")
    q.add_argument("--out", required=True, metavar="DIR",
                   help="results root; one <recording>.mea/ directory is written per recording")
    q.add_argument("--stage", metavar="STAGE",
                   help="run only this stage, entering from the file upstream of it "
                        "(qc_general | qc_crosstalk | qc_manual | integrate | quantify)")
    q.add_argument("--no-plots", action="store_true",
                   help="skip the diagnostic figures for this run "
                        "(they are on by default; plots.enabled turns them off "
                        "for good)")
    q.add_argument("--compile-out", metavar="DIR",
                   help="where the compiled tables go (default: <out>/compiled)")
    q.set_defaults(fn=cmd_run)

    # ---------------------------------------------------------- conditions
    q = sub.add_parser(
        "conditions", formatter_class=fmt,
        help="show which file is which condition",
        description="List each recording's condition files in order, with the index "
                    "an explicit `order` refers to and both clocks side by side.",
        epilog=textwrap.dedent("""\
            Run this before trusting an ordering.  A wrong condition order does not
            fail -- it relabels the drugs, and every downstream comparison then
            contrasts the wrong pair.

            The two clocks can disagree: `modified` belongs to the copy, `acquired`
            is read from inside the file and belongs to the experiment.

            To fix an order by hand, set in config:
              dataset:
                discovery:
                  condition_order: explicit
                  order: [4, 6, 1, 5, 2, 3]        # the # column below, 1-based
                  # or, safer, by name:
                  # order: [Baseline, Gabazine, CGP, D-AP5, DNQX, TTX]

            example:
              mea conditions --input /data/batch4
            """))
    q.add_argument("--input", nargs="+", required=True, metavar="DIR",
                   help="one or more directories to scan for recordings")
    q.set_defaults(fn=cmd_conditions)

    # ------------------------------------------------------------- compile
    q = sub.add_parser(
        "compile", formatter_class=fmt,
        help="join every recording's metrics into one table",
        description="The last stage. Concatenates what stage 6 measured for each "
                    "recording, labels each condition with its treatment, and "
                    "writes compiled.wide.csv (README.md, Metrics).",
        epilog=textwrap.dedent("""\
            Nothing is measured here -- it is a join, so it is cheap and repeatable.

            compiled.wide.csv has one row per recording and condition, one column
            per metric.

            --samples is a CSV with a `recording` column (the prefix) and any other
            columns -- genotype, line, batch, age.  They are added to each row
            after `recording`, so you can filter and group on them.

            Network values in a condition with fewer than 3 network events, or
            none (network.nBurst blank), are reported but rest on little.

            examples:
              mea compile --input results --out compiled
              mea compile --input results --out compiled --samples samples.csv
            """))
    q.add_argument("--input", required=True, metavar="DIR",
                   help="results root holding the .mea stores")
    q.add_argument("--out", required=True, metavar="DIR", help="where to write")
    q.add_argument("--samples", metavar="CSV",
                   help="per-recording columns (genotype, ...) to add to each row")
    q.set_defaults(fn=cmd_compile)

    # --------------------------------------------------------------- rawqc
    q = sub.add_parser(
        "rawqc", formatter_class=fmt,
        help="screen chips on the raw recording, before sorting",
        description="Count the electrodes TTX silences in each recording. A chip "
                    "with too few was never carrying a live organoid, and sorting "
                    "it is wasted time.",
        epilog=textwrap.dedent("""\
            Reads only the stored range headers, never the samples, so it takes about
            fifteen seconds per recording and does not need a sorter.

            Each recording is compared against ITS OWN untreated baseline. Using a
            drug condition as a stand-in inflates the active set and turns a failing
            chip into a passing one.

            The threshold is raw_qc.min_ttx_sensitive. On the lab's 14 example
            recordings every one filed as bad scores 14 or below and five of the six
            filed as good score 117 or above.

            examples:
              mea rawqc --input /data/batch4
              mea rawqc --input /data/batch4 --strict     # non-zero exit if any fail
            """))
    q.add_argument("--input", nargs="+", required=True, metavar="DIR",
                   help="one or more directories to scan for recordings")
    q.add_argument("--strict", action="store_true",
                   help="exit non-zero when any chip fails, for use in a script")
    q.set_defaults(fn=cmd_rawqc)

    # --------------------------------------------------------------- snake
    q = sub.add_parser(
        "snake", formatter_class=fmt,
        help="run via Snakemake: parallel across recordings, resumable",
        description="Hand the same work to Snakemake.  Recordings run concurrently "
                    "and re-running is a no-op for anything already finished.",
        epilog=textwrap.dedent("""\
            Requires the `snakemake` package.  State is tracked with sentinel files
            inside each .mea store, so interrupting and re-running resumes rather
            than restarting.

            examples:
              mea snake --input /data/batch4 --out results -j 8
              mea snake --input /data/batch4 --out results -n     # dry run: plan only
            """))
    q.add_argument("--input", nargs="+", required=True, metavar="DIR",
                   help="one or more directories to scan for recordings")
    q.add_argument("--out", required=True, metavar="DIR", help="results root")
    q.add_argument("-j", "--jobs", type=int, default=4, metavar="N",
                   help="recordings to run concurrently (default: 4)")
    q.add_argument("-n", "--dry-run", action="store_true",
                   help="show the plan without running anything")
    q.set_defaults(fn=cmd_snake)

    # -------------------------------------------------------------- import
    q = sub.add_parser(
        "import", aliases=["convert"], formatter_class=fmt,
        help="bring one legacy .mat into a .mea store",
        description="Read a single MATLAB .mat and write it as a .mea store, "
                    "reporting where in the pipeline it sits.  This is the first "
                    "step of `mea run`, exposed on its own for inspection.",
        epilog=textwrap.dedent("""\
            --out is the results ROOT, not a filename; the store is written to
            <out>/<recording>.mea/.

            Only files that still carry spike trains AND condition windows can be
            imported.  *_Analyzed.mat and *_Compiled.mat are results -- they drop
            rec_conditions -- so they are not valid inputs.

            example:
              mea import /data/batch4/344_..._curated_QCmanual.mat --out results
            """))
    q.add_argument("file", metavar="FILE", help="the .mat to import")
    q.add_argument("--out", required=True, metavar="DIR",
                   help="results root; the store is written to <DIR>/<recording>.mea/")
    q.set_defaults(fn=cmd_import)

    # Accept --config/--set after the command too; SUPPRESS keeps a top-level value
    for q in sub.choices.values():
        if "--config" not in q._option_string_actions:
            q.add_argument("--config", metavar="FILE", default=argparse.SUPPRESS,
                           help="YAML applied on top of config/default.yaml")
        if "--set" not in q._option_string_actions:
            q.add_argument("--set", action="append", metavar="KEY=VALUE",
                           default=argparse.SUPPRESS,
                           help="override one config value; repeatable")

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
