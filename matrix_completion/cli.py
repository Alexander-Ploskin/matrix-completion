from __future__ import annotations

import time
import cupy as cp
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import click
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from matrix_completion.dataset import MatrixGenerator, Matrix
from matrix_completion.logger import Logger

from matrix_completion.algorithms.rcg_matrix_completion import RCGMatrixCompletion
from matrix_completion.algorithms.lrg_geom_cg import LRGeomCG, p_omega
from matrix_completion.algorithms.simple_ls import SimpleLS


@dataclass(frozen=True)
class AlgoSpec:
    algo_id: str
    ctor: type
    base_params: Dict


def build_algos(tol: float, alpha: float) -> List[AlgoSpec]:
    return [
        AlgoSpec(
            algo_id="RCG_QPRECON",
            ctor=RCGMatrixCompletion,
            base_params={"num_iters": 3000, "tol": tol, "alpha": alpha,
                         "method": "rcg", "metric": "QPRECON"},
        ),
        AlgoSpec(
            algo_id="RCG_QRIGHTINV",
            ctor=RCGMatrixCompletion,
            base_params={"num_iters": 3000, "tol": tol, "alpha": alpha,
                         "method": "rcg", "metric": "QRIGHT-INV"},
        ),
        AlgoSpec(
            algo_id="RGD_QPRECON",
            ctor=RCGMatrixCompletion,
            base_params={"num_iters": 3000, "tol": tol, "alpha": alpha,
                         "method": "rgd", "metric": "QPRECON"},
        ),
        AlgoSpec(
            algo_id="RGD_QRIGHTINV",
            ctor=RCGMatrixCompletion,
            base_params={"num_iters": 3000, "tol": tol, "alpha": alpha,
                         "method": "rgd", "metric": "QRIGHT-INV"},
        ),
        AlgoSpec(
            algo_id="LRGeomCG",
            ctor=LRGeomCG,
            base_params={"num_iters": 500, "tol": tol, "singular_values_eps": 1e-6},
        ),
        AlgoSpec(
            algo_id="SimpleLS",
            ctor=SimpleLS,
            base_params={"num_iters": 500, "tol": tol, "lambda_reg": 0.0},
        ),
    ]


def observed_matrix(mat: Matrix) -> np.ndarray:
    # What algorithms should fit to (noisy observations if provided).
    if mat.M_noisy is not None:
        return mat.M_noisy
    return p_omega(mat.M_true, mat.mask)


def algo_legend_label(algo_id: str, rank: int, alpha: float) -> str:
    # MathText labels for legend (similar vibe as your example).
    # Keep them short so legend doesn’t cover plots.
    mapping = {
        "RCG_QPRECON":   r"$\mathrm{RCG}\;(\mathrm{QPRECON})$",
        "RCG_QRIGHTINV": r"$\mathrm{RCG}\;(\mathrm{QRIGHT\!-\!INV})$",
        "RGD_QPRECON":   r"$\mathrm{RGD}\;(\mathrm{QPRECON})$",
        "RGD_QRIGHTINV": r"$\mathrm{RGD}\;(\mathrm{QRIGHT\!-\!INV})$",
        "LRGeomCG":      r"$\mathrm{LRGeomCG}$",
        "SimpleLS":      r"$\mathrm{ALS/LS}$",
    }
    base = mapping.get(algo_id, algo_id)
    # Put parameters as formulas too:
    # - k is common in papers
    # - alpha only relevant for RCG/RGD (but harmless to show always)
    return f"{base} " + rf"$k={rank}$"


def safe_for_log(y: pd.Series, eps: float = 1e-16) -> pd.Series:
    # Avoid log(0); keep NaNs as NaNs
    return y.where(y.isna(), y.clip(lower=eps))


def run_one_algorithm(spec: AlgoSpec, matrix: Matrix, rank: int) -> Tuple[np.ndarray, pd.DataFrame]:
    params = dict(spec.base_params)
    params["rank"] = int(rank)

    model = spec.ctor(params=params)
    logger = Logger()

    click.echo(f"-> Running {spec.algo_id} (rank={rank})...")  # Click-friendly output. [web:81]
    X_hat = model.complete_matrix(matrix, logger=logger)

    df = pd.DataFrame(logger.records)
    if df.empty:
        # Hard fallback if algo didn’t log for some reason.
        Y = observed_matrix(matrix)
        Omega = matrix.mask
        df = pd.DataFrame([{
            "iter": 0,
            "time_s": float(getattr(logger, "elapsed_s", lambda: 0.0)()),
            "cost": np.nan,
            "grad_norm": np.nan,
            "dir_norm": np.nan,
            "rel_error": float(np.linalg.norm(X_hat - matrix.M_true) / np.linalg.norm(matrix.M_true)),
            "rel_residual": float(np.linalg.norm((Omega * (X_hat - Y))) / np.linalg.norm(Omega * Y)),
        }])

    df["algo"] = spec.algo_id
    return X_hat, df

def save_matrix_viz(mat: Matrix, X_hat: np.ndarray, out_dir: Path, prefix: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    M_true = mat.M_true
    M_obs = observed_matrix(mat)

    vmin = np.percentile(M_true, 1)
    vmax = np.percentile(M_true, 99)

    for name, M in [("true", M_true), ("observed", M_obs), ("completed", X_hat)]:
        if hasattr(M, 'get'):
            M = M.get()
        plt.figure(figsize=(6, 5))
        sns.heatmap(M, cmap="viridis", vmin=vmin, vmax=vmax)
        plt.title(f"{prefix}: {name}")
        plt.tight_layout()
        plt.savefig(out_dir / f"{prefix}_{name}.png", dpi=200)
        plt.close()

def plot_metrics_single_algo(df_algo: pd.DataFrame, out_path: Path, title: str, legend_label: str) -> None:
    sns.set_theme(style="darkgrid")

    metrics = [
        ("rel_error",     r"$\|X - A\|_F / \|A\|_F$",                         "Reconstruction Error Over Iterations"),
        ("grad_norm",     r"$\|\nabla f\|_F$",                                 "Gradient Norm Over Iterations"),
        ("rel_residual",  r"$\|P_{\Omega}(X - Y)\|_F / \|P_{\Omega}(Y)\|_F$",  "Reconstruction Residual Over Iterations"),
        ("dir_norm",      r"$\|\eta\|_F$",                                     "Conjugate Direction Norm Over Iterations"),
    ]

    df_algo = df_algo.sort_values("iter").copy()
    fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 12), sharex=True)

    x = df_algo["iter"].to_numpy()

    for ax, (col, ylabel, subtitle) in zip(axes, metrics):
        ax.set_title(subtitle)
        ax.set_ylabel(ylabel)
        ax.set_yscale("log")  # log scale y-axis [web:222]

        if col not in df_algo.columns or not df_algo[col].notna().any():
            # Keep subplot non-empty (LS: grad_norm/dir_norm), but make it explicit.
            y_placeholder = np.full_like(x, 1.0, dtype=float)
            sns.lineplot(ax=ax, x=x, y=y_placeholder, label=legend_label, errorbar=None)
            ax.lines[-1].set_linestyle("--")
            ax.lines[-1].set_alpha(0.35)

            # Annotation instead of turning axis off. [web:242]
            ax.text(
                0.5, 0.5, f"{col}: N/A for this method",
                transform=ax.transAxes,
                ha="center", va="center",
            )  # [web:242]

            ax.set_ylim(1e-3, 1e3)  # stable log window for the placeholder
            ax.legend(loc="upper right", frameon=True)
            continue

        y = df_algo[col].copy()
        y = y.where(y.isna(), y.clip(lower=1e-16))  # avoid log(0)

        sns.lineplot(ax=ax, x=df_algo["iter"], y=y, label=legend_label, errorbar=None)  # [web:15]
        ax.legend(loc="upper right", frameon=True)

    axes[-1].set_xlabel("Iteration")
    fig.suptitle(title, y=0.995)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


@click.command()
@click.option("--m", type=int, default=900, show_default=True)
@click.option("--n", type=int, default=900, show_default=True)
@click.option("--rank", type=int, default=10, show_default=True)
@click.option("--os", "OS", type=float, default=0.9, show_default=True, help="Missing fraction (as in your generator).")

@click.option("--noise-level", type=float, default=0.0, show_default=True)
@click.option("--seed", type=int, default=42, show_default=True)

@click.option("--alpha", type=float, default=0.33, show_default=True)
@click.option("--num-iters", type=int, default=1000, show_default=True)
@click.option("--tol", type=float, default=1e-6, show_default=True)

@click.option("--out", "out_dir", type=click.Path(path_type=Path), default=Path("out"), show_default=True)

@click.option("--print-matrices/--no-print-matrices", default=False, show_default=True)
@click.option("--print-limit", type=int, default=20, show_default=True, help="If matrix bigger than this, print top-left block.")
def run(m, n, rank, OS, noise_level, seed, alpha, num_iters, tol, out_dir, print_matrices, print_limit):
    """
    Run all methods once for a single environment and save ONE multi-panel PNG with curves per method.
    """
    click.secho("Starting evaluation (single environment)...", fg="green")  # Click styling helpers. [web:81]
    click.echo(f"m={m}, n={n}, rank={rank}, OS={OS}, noise_level={noise_level}, seed={seed}")
    click.echo(f"num_iters={num_iters}, tol={tol}, alpha={alpha}")
    out_dir.mkdir(parents=True, exist_ok=True)

    MG = MatrixGenerator()
    algos = build_algos(tol=tol, alpha=alpha)

    click.echo("Generating matrix...")
    mat = MG.get_matrix(
        m=m, n=n, k=int(rank),
        missing_fraction=float(OS),
        noise_level=float(noise_level),
        random_state=int(seed),
    )

    click.echo(f"Running {len(algos)} algorithms...")
    dfs: List[pd.DataFrame] = []

    # Optional: keep one completed matrix to print (e.g., last algo)
    last_X = None

    stamp = time.strftime("%Y%m%d_%H%M%S")
    env_tag = f"m{m}_n{n}_k{rank}_OS{OS}_noise{noise_level}_seed{seed}_{stamp}"

    for spec in algos:
        cp.get_default_pinned_memory_pool().free_all_blocks()
        X_hat, df = run_one_algorithm(spec, mat, rank=rank)

        # save per-algo PNG containing all metrics as subplots
        algo_png = out_dir / f"{spec.algo_id}__metrics__{env_tag}.png"
        legend = algo_legend_label(spec.algo_id, rank=rank, alpha=alpha)  # your mathtext label
        title = f"{spec.algo_id}: m={m}, n={n}, k={rank}, OS={OS}, noise={noise_level}, seed={seed}"
        plot_metrics_single_algo(df, algo_png, title=title, legend_label=legend)
        click.echo(f"Saved: {algo_png}")

        if print_matrices:
            save_matrix_viz(mat, X_hat, out_dir / f"{spec.algo_id}__matrices__{env_tag}", prefix=spec.algo_id)
            click.echo(f"Saved matrices to: {out_dir / f'{spec.algo_id}__matrices__{env_tag}'}")

if __name__ == "__main__":
    run()
