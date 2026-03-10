import argparse
import os

import apple_bolt as bolt
import matplotlib.pyplot as plt
import yaml


def collect_metrics(bolt_id, prefix):
    task = bolt.get_task(bolt_id)
    all_metrics = task.get_metrics()[bolt_id]
    return filter_metrics_by_prefix(all_metrics, prefix)


def filter_metrics_by_prefix(all_metrics, prefix):
    """Filter metrics by prefix and strip it from the keys."""
    prefix_slash = f"{prefix}/"

    filtered_metrics = {k.replace(prefix_slash, ""): list(all_metrics[k].metric_value) for k in all_metrics.keys() if prefix in k}

    return filtered_metrics


def load_metrics_from_yaml(yaml_path):
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f)


def dump_metrics_to_yaml(metrics, output_path):
    with open(output_path, "w") as f:
        yaml.dump(metrics, f, default_flow_style=False)
    print(f"Metrics saved to {output_path}")


def generate_paper_figures(metrics, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    max_epoch = max(metrics["epoch"])

    def epoch_axis(values):
        """Generate evenly spaced epoch values from 0 to max_epoch."""
        n = len(values)
        return [i * max_epoch / (n - 1) for i in range(n)] if n > 1 else [0]

    train_loss = metrics["loss"]
    eval_loss = metrics["eval_validation_clean_loss"]

    # Figure 1: Training and Validation Loss
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epoch_axis(train_loss), train_loss, label="Training Loss", alpha=0.7)
    ax.plot(epoch_axis(eval_loss), eval_loss, label="Validation Loss", marker="s", markersize=3)
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training and Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    loss_path = os.path.join(output_dir, "loss.png")
    fig.savefig(loss_path, dpi=300)
    plt.close(fig)
    print(f"Loss figure saved to {loss_path}")

    # Figure 2: WER over training (4 splits)
    wer_series = [
        ("dev-clean", metrics.get("eval_validation_clean_wer", []), "o", "--"),
        ("dev-other", metrics.get("eval_validation_other_wer", []), "s", "--"),
        ("test-clean", metrics.get("eval_test_clean_wer", []), "^", "-"),
        ("test-other", metrics.get("eval_test_other_wer", []), "D", "-"),
    ]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, (label, values, marker, linestyle) in enumerate(wer_series):
        if values:
            values_pct = [v * 100 for v in values]
            min_val = min(values_pct)
            color = colors[i % len(colors)]
            ax.plot(epoch_axis(values_pct), values_pct, label=f"{label} (min: {min_val:.2f}%)",
                    marker=marker, markersize=3, linestyle=linestyle, color=color)
            ax.axhline(y=min_val, color=color, linestyle=":", alpha=0.4, linewidth=0.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("WER (%)")
    ax.set_title("Word Error Rate over Training (Greedy Decoding)")
    ax.legend(fontsize="small")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    wer_path = os.path.join(output_dir, "wer.png")
    fig.savefig(wer_path, dpi=300)
    plt.close(fig)
    print(f"WER figure saved to {wer_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect Bolt metrics and generate paper figures")
    parser.add_argument("--bolt-id", default="4rkpchk6wq", help="Bolt task ID")
    parser.add_argument("--prefix", default=None, help="Metric prefix to filter by (e.g. libri-100h-eval-other-run-2)")
    parser.add_argument("--from-yaml", type=str, default="bolt_metrics_libri-100h-eval-other-run-2.yaml", help="Load metrics from a YAML file instead of Bolt")
    parser.add_argument("--dump-yaml", action="store_true", help="Dump metrics to YAML file")
    parser.add_argument("--output-dir", default="project/docs", help="Directory for generated figures")
    args = parser.parse_args()

    if args.from_yaml:
        metrics = load_metrics_from_yaml(args.from_yaml)
    else:
        metrics = collect_metrics(args.bolt_id, args.prefix)

    if args.dump_yaml:
        dump_metrics_to_yaml(metrics, f"bolt_metrics_{args.prefix}.yaml")

    print(f"Available metrics: {list(metrics.keys())}")
    generate_paper_figures(metrics, args.output_dir)