"""Evaluate a wake word model and produce a DET curve plot."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort

from ..config import WakeWordConfig

logger = logging.getLogger(__name__)


def _load_validation_features(config: WakeWordConfig) -> tuple[np.ndarray, np.ndarray]:
    """Load positive and negative validation features from pre-extracted .npy files.

    Returns:
        (positive_features, negative_features) each shaped (N, 16, 96).
    """
    model_dir = config.model_output_dir
    pos_path = model_dir / "positive_features_test.npy"
    neg_path = model_dir / "negative_features_test.npy"

    pos = np.load(str(pos_path)) if pos_path.exists() else np.zeros((0, 16, 96))
    neg = np.load(str(neg_path)) if neg_path.exists() else np.zeros((0, 16, 96))

    # Also include background noise test features if available
    bg_test_path = model_dir / "background_noise_features_test.npy"
    if bg_test_path.exists():
        bg_neg = np.load(str(bg_test_path))
        neg = np.concatenate([neg, bg_neg], axis=0) if neg.shape[0] > 0 else bg_neg

    # Also include general negative validation features if available
    val_path = config.data_path / "features" / "validation_set_features.npy"
    if val_path.exists():
        val_neg = np.load(str(val_path))
        if val_neg.ndim == 2:
            n_full = (val_neg.shape[0] // 16) * 16
            remainder = val_neg.shape[0] - n_full
            if remainder > 0:
                logger.warning(
                    "Dropping %d/%d validation samples (not divisible by 16)",
                    remainder,
                    val_neg.shape[0],
                )
            val_neg = val_neg[:n_full].reshape(-1, 16, 96)
        neg = np.concatenate([neg, val_neg], axis=0) if neg.shape[0] > 0 else val_neg

    if pos.shape[0] == 0:
        raise ValueError(
            f"No positive validation features found at {pos_path}. "
            "Run the generate/augment pipeline first."
        )
    if neg.shape[0] == 0:
        raise ValueError(
            f"No negative validation features found. "
            f"Checked {neg_path} and {val_path}. "
            "Run setup and the generate/augment pipeline first."
        )

    logger.info(f"Loaded {pos.shape[0]} positive, {neg.shape[0]} negative validation samples")
    return pos, neg


def _predict_onnx(
    session: ort.InferenceSession,
    features: np.ndarray,
    batch_size: int = 1,
) -> np.ndarray:
    """Run ONNX model on feature batches, return scores array."""
    input_name = session.get_inputs()[0].name
    all_scores: list[np.ndarray] = []
    for i in range(0, len(features), batch_size):
        batch = features[i : i + batch_size].astype(np.float32)
        outputs = session.run(None, {input_name: batch})
        all_scores.append(outputs[0].squeeze(-1))
    return np.concatenate(all_scores, axis=0)


def _compute_det_curve(
    pos_scores: np.ndarray,
    neg_scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute DET curve points (FPR, FNR) across thresholds.

    Returns:
        (thresholds, fpr, fnr) arrays sorted by ascending threshold.
    """
    thresholds = np.linspace(0.0, 1.0, 1001)
    fpr = np.array([np.mean(neg_scores >= t) for t in thresholds])
    fnr = np.array([np.mean(pos_scores < t) for t in thresholds])
    return thresholds, fpr, fnr


def _compute_aut(fpr: np.ndarray, fnr: np.ndarray) -> float:
    """Compute Area Under the DET curve (AUT) using the trapezoidal rule.

    Lower is better (0 = perfect).
    We integrate FNR as a function of FPR (sorted by ascending FPR).
    """
    # Sort by FPR for proper integration
    sort_idx = np.argsort(fpr)
    fpr_sorted = fpr[sort_idx]
    fnr_sorted = fnr[sort_idx]
    return float(np.trapezoid(fnr_sorted, fpr_sorted))


def _plot_det_curve(
    fpr: np.ndarray,
    fnr: np.ndarray,
    aut: float,
    model_name: str,
    output_path: Path,
    metrics: dict[str, Any],
) -> None:
    """Render DET curve to PNG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    fig, ax = plt.subplots(figsize=(8, 7))

    # Plot DET curve (FPR vs FNR)
    sort_idx = np.argsort(fpr)
    ax.plot(fpr[sort_idx] * 100, fnr[sort_idx] * 100, linewidth=2, color="#2563eb")

    # Shade AUT area
    ax.fill_between(
        fpr[sort_idx] * 100,
        fnr[sort_idx] * 100,
        alpha=0.15,
        color="#2563eb",
    )

    ax.set_xlabel("False Positive Rate (%)", fontsize=13)
    ax.set_ylabel("False Negative Rate (%)", fontsize=13)
    ax.set_title(f"DET Curve \u2014 {model_name}", fontsize=15, fontweight="bold")

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f"))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f"))
    ax.grid(True, alpha=0.3)

    # Diagonal reference (random classifier)
    ax.plot([0, 100], [100, 0], "--", color="gray", alpha=0.5, label="Random")

    # Annotation box with metrics
    if "optimal_thresholds" in metrics:
        threshold_lines = [
            f"Hit {i}: {threshold:.2f}"
            for i, threshold in enumerate(metrics["optimal_thresholds"], start=1)
        ]
    else:
        threshold_lines = [f"Optimal Thresh: {metrics['optimal_threshold']:.2f}"]

    text_lines = [
        f"AUT: {aut:.4f}",
        f"FPPH: {metrics['fpph']:.2f}",
        f"Recall: {metrics['recall']:.1%}",
        f"Threshold: {metrics['threshold']:.2f}",
        *threshold_lines,
    ]
    ax.text(
        0.97,
        0.97,
        "\n".join(text_lines),
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="#ccc", alpha=0.9),
        fontfamily="monospace",
    )

    ax.legend(loc="lower left", fontsize=10)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)
    logger.info(f"DET curve saved to {output_path}")


def _find_hit_thresholds(
    pos_scores: np.ndarray,
    neg_scores: np.ndarray,
    validation_hours: float,
    target_fpph: float,
    hit_count: int,
) -> list[dict[str, Any]]:
    """Greedily optimize one threshold for each required hit.

    Each hit threshold is optimized on the examples that survived the previous
    hit threshold. This models a multi-hit detector where later hits only see
    candidates that crossed earlier hit gates.
    """
    from ..training.metrics import find_best_threshold

    if hit_count < 1:
        raise ValueError("hit_count must be >= 1")

    stages: list[dict[str, Any]] = []
    stage_pos = pos_scores
    stage_neg = neg_scores
    original_pos_count = len(pos_scores)
    effective_threshold = 0.0

    for hit_index in range(1, hit_count + 1):
        optimal = find_best_threshold(
            stage_pos,
            stage_neg,
            validation_hours=validation_hours,
            target_fpph=target_fpph,
        )
        threshold = optimal["threshold"]
        effective_threshold = max(effective_threshold, threshold)
        cumulative_pos_mask = pos_scores >= effective_threshold
        cumulative_neg_mask = neg_scores >= effective_threshold
        cumulative_recall = float(np.mean(cumulative_pos_mask)) if original_pos_count > 0 else 0.0
        cumulative_fpph = (
            float(np.sum(cumulative_neg_mask) / validation_hours)
            if validation_hours > 0
            else float("inf")
        )
        stages.append(
            {
                "hit": hit_index,
                "threshold": threshold,
                "recall": optimal["recall"],
                "fpph": optimal["fpph"],
                "accuracy": optimal["accuracy"],
                "cumulative_recall": cumulative_recall,
                "cumulative_fpph": cumulative_fpph,
                "n_positive": float(len(stage_pos)),
                "n_negative": float(len(stage_neg)),
            }
        )
        stage_pos = stage_pos[stage_pos >= threshold]
        stage_neg = stage_neg[stage_neg >= threshold]

    return stages


def run_eval(
    config: WakeWordConfig,
    model_path: str | Path,
    hit_count: int | None = None,
) -> dict[str, Any]:
    """Run full evaluation: compute scores, DET curve, AUT, and save plot + metrics JSON.

    Args:
        config: Wake word configuration (used to locate validation data).
        model_path: Path to the ONNX classifier model to evaluate.
        hit_count: Number of detection hits to optimize thresholds for. Defaults
            to config.eval_hit_count.

    Returns:
        Dict with keys: aut, fpph, recall, accuracy, threshold
    """
    resolved_hit_count = config.eval_hit_count if hit_count is None else hit_count
    if resolved_hit_count < 1:
        raise ValueError("hit_count must be >= 1")

    # Load ONNX model
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    logger.info(f"Loaded model from {model_path}")

    # Load validation data
    pos_features, neg_features = _load_validation_features(config)

    # Run predictions
    logger.info("Running predictions on validation set...")
    pos_scores = _predict_onnx(session, pos_features)
    neg_scores = _predict_onnx(session, neg_features)

    # Compute DET curve
    thresholds, fpr, fnr = _compute_det_curve(pos_scores, neg_scores)

    # Compute AUT
    aut = _compute_aut(fpr, fnr)

    # Compute summary metrics at fixed threshold 0.5 for consistent comparison
    clip_duration = config.augmentation.clip_duration
    validation_hours = neg_features.shape[0] * clip_duration / 3600.0

    from ..training.metrics import evaluate_model

    fixed = evaluate_model(
        pos_scores,
        neg_scores,
        threshold=0.5,
        validation_hours=validation_hours,
    )

    hit_thresholds = _find_hit_thresholds(
        pos_scores=pos_scores,
        neg_scores=neg_scores,
        validation_hours=validation_hours,
        target_fpph=config.target_fp_per_hour,
        hit_count=resolved_hit_count,
    )
    optimal = hit_thresholds[-1]
    optimal_thresholds = [stage["threshold"] for stage in hit_thresholds]

    # Build results
    results: dict[str, Any] = {
        "aut": aut,
        "fpph": fixed["fpph"],
        "recall": fixed["recall"],
        "accuracy": fixed["accuracy"],
        "threshold": fixed["threshold"],
        "optimal_threshold": optimal["threshold"],
        "optimal_recall": optimal["recall"],
        "optimal_fpph": optimal["fpph"],
        "hit_count": resolved_hit_count,
        "optimal_thresholds": optimal_thresholds,
        "hit_thresholds": hit_thresholds,
        "n_positive": int(pos_features.shape[0]),
        "n_negative": int(neg_features.shape[0]),
        "validation_hours": round(validation_hours, 2),
    }

    # Save plot
    output_dir = config.model_output_dir
    plot_path = output_dir / f"{config.model_name}_det.png"
    plot_metrics = {
        **fixed,
        "optimal_threshold": optimal["threshold"],
        "optimal_thresholds": optimal_thresholds,
    }
    _plot_det_curve(fpr, fnr, aut, config.model_name, plot_path, plot_metrics)

    # Save metrics JSON
    metrics_path = output_dir / f"{config.model_name}_eval.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(results, indent=2) + "\n")
    logger.info(f"Eval metrics saved to {metrics_path}")

    return results
