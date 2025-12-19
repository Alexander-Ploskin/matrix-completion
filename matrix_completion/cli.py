from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import click
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm

from matrix_completion.dataset import MatrixGenerator, Matrix
from matrix_completion.utils.metrics import (
    calculate_relative_error,
    calculate_relative_residual,
)
from matrix_completion.logger import Logger

from matrix_completion.algorithms.rcg_matrix_completion import RCGMatrixCompletion
from matrix_completion.algorithms.lrg_geom_cg import LRGeomCG, p_omega
from matrix_completion.algorithms.simple_ls import SimpleLS


@dataclass(frozen=True)
class AlgoSpec:
    algo_id: str
    ctor: type
    base_params: Dict


def build_algos(num_iters: int, tol: float, alpha: float) -> List[AlgoSpec]:
    return [
        AlgoSpec(
            algo_id="RCG_QPRECON",
            ctor=RCGMatrixCompletion,
            base_params={"num_iters": num_iters, "tol": tol, "alpha": alpha,
                         "method": "rcg", "metric": "QPRECON"},
        ),
        AlgoSpec(
            algo_id="RGD_QPRECON",
            ctor=RCGMatrixCompletion,
            base_params={"num_iters": num_iters, "tol": tol, "alpha": alpha,
                         "method": "rgd", "metric": "QPRECON"},
        ),
        AlgoSpec(
            algo_id="LRGeomCG",
            ctor=LRGeomCG,
            base_params={"num_iters": num_iters, "tol": tol, "singular_values_eps": 1e-6},
        ),
        AlgoSpec(
            algo_id="SimpleLS",
            ctor=SimpleLS,
            base_params={"num_iters": num_iters, "tol": tol},
        ),
    ]


def observed_matrix(mat: Matrix) -> np.ndarray:
    if mat.M_noisy is not None:
        return mat.M_noisy
    return p_omega(mat.M_true, mat.mask)


def run_one_algorithm(
    spec: AlgoSpec,
    matrix: Matrix,
    rank: int,
    meta: Dict | None = None,
) -> Tuple[np.ndarray, pd.DataFrame, Dict]:
    meta = meta or {}

    params = dict(spec.base_params)
    params["rank"] = rank
    model = spec.ctor(params=params)

    # Logger owns timing + per-iter metrics
    logger = Logger(meta=meta)
    tqdm.write(f"  -> Running {spec.algo_id} (rank={rank})...")  # safe with tqdm [web:88]
    X_hat = model.complete_matrix(matrix, logger=logger)

    df = pd.DataFrame(logger.records)

    if df.empty:
        # Ensure at least one record exists
        M_obs = observed_matrix(matrix)
        Omega = matrix.mask
        df = pd.DataFrame([{
            "iter": 0,
            "time_s": float(logger.elapsed_s()),
            "rel_error": float(calculate_relative_error(X_hat, matrix.M_true)),
            "rel_residual": float(calculate_relative_residual(X_hat, M_obs, Omega)),
            "grad_norm": np.nan,
            "dir_norm": np.nan,
            "cost": np.nan,
            **meta,
        }])

    summary = {
        "algo": spec.algo_id,
        "rank": rank,
        "iters": int(df["iter"].max()),
        "time_s": float(df["time_s"].max()) if "time_s" in df.columns else np.nan,
        "final_rel_error": float(df["rel_error"].dropna().iloc[-1]),
        "final_rel_residual": float(df["rel_residual"].dropna().iloc[-1]),
        "final_grad_norm": float(df["grad_norm"].dropna().iloc[-1]) if df["grad_norm"].notna().any() else np.nan,
        "final_dir_norm": float(df["dir_norm"].dropna().iloc[-1]) if df["dir_norm"].notna().any() else np.nan,
        **meta,
    }

    tqdm.write(
        f"  <- Done {spec.algo_id}: "
        f"time={summary['time_s']:.3f}s, "
        f"final_rel_error={summary['final_rel_error']:.3e}, "
        f"final_rel_residual={summary['final_rel_residual']:.3e}"
    )

    return X_hat, df, summary


def save_matrix_viz(mat: Matrix, X_hat: np.ndarray, out_dir: Path, prefix: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    M_true = mat.M_true
    M_obs = observed_matrix(mat)

    vmin = np.percentile(M_true, 1)
    vmax = np.percentile(M_true, 99)

    for name, M in [("true", M_true), ("observed", M_obs), ("completed", X_hat)]:
        if hasattr(M, 'get'): # came from cupy
            M = M.get()
        plt.figure(figsize=(6, 5))
        sns.heatmap(M, cmap="viridis", vmin=vmin, vmax=vmax)
        plt.title(f"{prefix}: {name}")
        plt.tight_layout()
        plt.savefig(out_dir / f"{prefix}_{name}.png", dpi=200)
        plt.close()


def plot_curves(df_iters: pd.DataFrame, out_dir: Path, env_tag: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="darkgrid")

    def _plot(metric: str, x: str, logy: bool = False):
        plt.figure(figsize=(8, 5))
        ax = sns.lineplot(data=df_iters, x=x, y=metric, hue="algo", style="algo", errorbar=None)
        if logy:
            ax.set_yscale("log")
        plt.title(f"{env_tag}: {metric} vs {x}")
        plt.tight_layout()
        plt.savefig(out_dir / f"{env_tag}__{metric}_vs_{x}.png", dpi=200)
        plt.close()

    _plot("rel_error", "iter", logy=True)
    _plot("grad_norm", "iter", logy=True)
    _plot("rel_residual", "iter", logy=True)
    _plot("dir_norm", "iter", logy=True)

    _plot("rel_error", "time_s", logy=True)
    _plot("grad_norm", "time_s", logy=True)
    _plot("rel_residual", "time_s", logy=True)
    _plot("dir_norm", "time_s", logy=True)


def ensure_list(ctx, param, value):
    if value is None:
        return []
    if len(value) == 1 and isinstance(value[0], str) and "," in value[0]:
        return [float(x) if "." in x else int(x) for x in value[0].split(",")]
    return list(value)


@click.group()
def mc():
    """Matrix completion evaluation CLI."""
    pass


@mc.command()
@click.option("--m", type=int, default=900, show_default=True)
@click.option("--n", type=int, default=900, show_default=True)
@click.option("--ranks", multiple=True, callback=ensure_list, default=("10,50",), show_default=True)
@click.option("--os", "OSs", multiple=True, callback=ensure_list, default=("0.3,0.7,0.9",), show_default=True)
@click.option("--noise-level", type=float, default=0.0, show_default=True)
@click.option("--seed", type=int, default=42, show_default=True)

@click.option("--alpha", type=float, default=0.33, show_default=True)
@click.option("--num-iters", type=int, default=30_000, show_default=True)
@click.option("--tol", type=float, default=1e-6, show_default=True)

@click.option("--out", "out_dir", type=click.Path(path_type=Path), default=Path("out"), show_default=True)
@click.option("--viz/--no-viz", default=True, show_default=True)
def run(m, n, ranks, OSs, noise_level, seed, alpha, num_iters, tol, out_dir, viz):
    click.secho("Starting run evaluation...", fg="green")  # Click supports styled output [web:81]
    click.echo(f"Grid: ranks={list(ranks)} OSs={list(OSs)}")
    click.echo(f"Matrix: m={m}, n={n}, noise_level={noise_level}, seed={seed}")
    click.echo(f"Optimization: num_iters={num_iters}, tol={tol}, alpha={alpha}")
    click.echo(f"Viz: {viz}")
    click.echo(f"Output root: {out_dir}")

    MG = MatrixGenerator()
    algos = build_algos(num_iters=num_iters, tol=tol, alpha=alpha)

    run_dir = out_dir / time.strftime("run_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    click.echo(f"Run directory: {run_dir}")

    all_iters: List[pd.DataFrame] = []
    all_summaries: List[Dict] = []

    for rank in ranks:
        for OS in OSs:
            click.secho(f"\nGenerating matrix for rank={int(rank)} OS={float(OS)} ...", fg="cyan")

            mat = MG.get_matrix(
                m=m, n=n, k=int(rank),
                missing_fraction=float(OS),
                noise_level=float(noise_level),
                random_state=int(seed),
            )

            env_tag = f"m{m}_n{n}_rank{int(rank)}_OS{float(OS)}_noise{noise_level}_seed{seed}"
            env_dir = run_dir / env_tag
            env_dir.mkdir(parents=True, exist_ok=True)
            click.echo(f"Environment dir: {env_dir}")

            click.echo(f"Running {len(algos)} algorithms...")
            for spec in algos:
                meta = {
                    "algo": spec.algo_id,
                    "rank": int(rank),
                    "OS": float(OS),
                    "noise_level": float(noise_level),
                    "seed": int(seed),
                    "m": int(m),
                    "n": int(n),
                }

                X_hat, df, summary = run_one_algorithm(spec, mat, rank=int(rank), meta=meta)
                all_summaries.append(summary)
                all_iters.append(df)

                if viz:
                    click.echo(f"Saving matrix visualizations for {spec.algo_id}...")
                    save_matrix_viz(mat, X_hat, env_dir / "viz", prefix=spec.algo_id)

            click.echo("Plotting curves for this environment...")
            df_env = pd.concat([d for d in all_iters if (d["rank"].iloc[0] == int(rank) and d["OS"].iloc[0] == float(OS))],
                               ignore_index=True)
            plot_curves(df_env, env_dir / "plots", env_tag=env_tag)
            click.echo(f"Saved plots to: {env_dir / 'plots'}")

    click.echo("\nSaving aggregated CSVs...")
    df_iters = pd.concat(all_iters, ignore_index=True)
    df_summary = pd.DataFrame(all_summaries)

    df_iters.to_csv(run_dir / "iters.csv", index=False)
    df_summary.to_csv(run_dir / "summary.csv", index=False)

    click.echo("Saving run config...")
    (run_dir / "config.json").write_text(json.dumps({
        "m": m, "n": n, "ranks": list(ranks), "OSs": list(OSs),
        "noise_level": noise_level, "seed": seed,
        "alpha": alpha, "num_iters": num_iters, "tol": tol,
        "viz": viz,
    }, indent=2))

    click.secho(f"Done. Saved results to: {run_dir}", fg="green")


@mc.command()
@click.option("--m", type=int, default=900, show_default=True)
@click.option("--n", type=int, default=900, show_default=True)
@click.option("--rank", type=int, default=10, show_default=True)
@click.option("--os", "OS", type=float, default=0.9, show_default=True)
@click.option("--noise-level", type=float, default=0.0, show_default=True)

@click.option("--trials", type=int, default=10, show_default=True)
@click.option("--seed0", type=int, default=42, show_default=True)

@click.option("--alpha", type=float, default=0.33, show_default=True)
@click.option("--num-iters", type=int, default=30_000, show_default=True)
@click.option("--tol", type=float, default=1e-6, show_default=True)

@click.option("--out", "out_dir", type=click.Path(path_type=Path), default=Path("out"), show_default=True)
@click.option("--save-curves/--no-save-curves", default=False, show_default=True)
def sweep(m, n, rank, OS, noise_level, trials, seed0, alpha, num_iters, tol, out_dir, save_curves):
    click.secho("Starting sweep evaluation...", fg="green")
    click.echo(f"Env: m={m}, n={n}, rank={rank}, OS={OS}, noise_level={noise_level}")
    click.echo(f"Trials: {trials} (seed0={seed0})")
    click.echo(f"Save curves: {save_curves}")
    click.echo(f"Output root: {out_dir}")

    MG = MatrixGenerator()
    algos = build_algos(num_iters=num_iters, tol=tol, alpha=alpha)

    run_dir = out_dir / time.strftime("sweep_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    click.echo(f"Sweep directory: {run_dir}")

    summaries = []
    curves = []

    for t in range(trials):
        seed = seed0 + t
        click.secho(f"\nTrial {t+1}/{trials}: generating matrix (seed={seed})...", fg="cyan")

        mat = MG.get_matrix(
            m=m, n=n, k=rank,
            missing_fraction=OS,
            noise_level=noise_level,
            random_state=seed,
        )

        click.echo(f"Running {len(algos)} algorithms...")
        for spec in algos:
            meta = {
                "algo": spec.algo_id,
                "trial": int(t),
                "seed": int(seed),
                "m": int(m),
                "n": int(n),
                "rank": int(rank),
                "OS": float(OS),
                "noise_level": float(noise_level),
            }
            _, df, summary = run_one_algorithm(spec, mat, rank=rank, meta=meta)
            summaries.append(summary)

            if save_curves:
                curves.append(df)

    click.echo("\nSaving sweep CSVs...")
    df_sum = pd.DataFrame(summaries)
    df_sum.to_csv(run_dir / "summary_raw.csv", index=False)

    click.echo("Aggregating statistics...")
    agg = (
        df_sum.groupby("algo")
        .agg(
            trials=("trial", "count"),
            mean_time_s=("time_s", "mean"),
            std_time_s=("time_s", "std"),
            mean_final_rel_error=("final_rel_error", "mean"),
            std_final_rel_error=("final_rel_error", "std"),
            mean_final_rel_residual=("final_rel_residual", "mean"),
            std_final_rel_residual=("final_rel_residual", "std"),
        )
        .reset_index()
    )
    agg.to_csv(run_dir / "summary_stats.csv", index=False)

    if save_curves and curves:
        click.echo("Saving per-iteration curves for all trials...")
        pd.concat(curves, ignore_index=True).to_csv(run_dir / "iters_all_trials.csv", index=False)

    click.secho(f"Done. Saved sweep results to: {run_dir}", fg="green")


@mc.command()
@click.argument("iters_csv", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--out", "out_dir", type=click.Path(path_type=Path), default=Path("out_plots"), show_default=True)
def plot(iters_csv: Path, out_dir: Path):
    click.secho("Plotting from CSV...", fg="green")
    click.echo(f"Input: {iters_csv}")
    click.echo(f"Output: {out_dir}")

    df = pd.read_csv(iters_csv)
    out_dir.mkdir(parents=True, exist_ok=True)

    keys = ["rank", "OS", "seed", "noise_level"]
    click.echo(f"Found {df.groupby(keys).ngroups} environments in CSV.")
    for env_vals, df_env in df.groupby(keys):
        env_tag = "_".join(f"{k}{v}" for k, v in zip(keys, env_vals))
        click.echo(f"Plotting env: {env_tag}")
        plot_curves(df_env, out_dir, env_tag=env_tag)

    click.secho("Done plotting.", fg="green")


def main():
    mc()


if __name__ == "__main__":
    main()
