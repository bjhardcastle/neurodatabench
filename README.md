# NeuroDataBench

Benchmark artifacts are written to the directory passed as `out` to `neurodatabench.main()`. The `--out` CLI option and `NDB_OUT` environment variable can override it. If none is supplied, artifacts are written directly beside the implementation file.

## Benchmark runs

Run the currently configured benchmark matrix into a results directory:

```powershell
uv run python scripts/run_benchmark_matrix.py --out benchmark-runs-20260909-dev6\results
```

This appends new timestamped run directories. It does not replay the bespoke dev6/true-PyNWB six-run set in that directory; the checked-in matrix currently uses different lazynwb rows.

Regenerate each completed run's `dashboard.html` and the aggregate leaderboard:

```powershell
uv run python -c "import json; from pathlib import Path; import neurodatabench.models as m; import neurodatabench.plots as p; import neurodatabench.runner as r; root=Path(r'benchmark-runs-20260909-dev6\results'); [p._write_result_dashboard(out_dir=d, metadata=json.loads((d/'run_metadata.json').read_text()), timings=m.RunTimings.model_validate(t), profile_samples=[json.loads(x) for x in (d/'profile_samples.jsonl').read_text().splitlines()], profile_summary=json.loads((d/'profile_summary.json').read_text()), run_start_wall_time_ns=int(json.loads((d/'profile_samples.jsonl').read_text().splitlines()[0])['time_ns'])) for d in root.iterdir() if d.is_dir() for t in [json.loads((d/'timings.json').read_text())] if 'setup_duration_ns' in t]; r._update_results_leaderboard(next(d for d in root.iterdir() if d.is_dir()))"
```

Timed-out runs are included in the aggregate leaderboard but do not have enough persisted timing data to regenerate an individual dashboard.
