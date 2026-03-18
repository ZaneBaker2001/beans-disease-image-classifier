import numpy as np
import matplotlib.pyplot as plt


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == labels).astype(np.float32)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(labels)

    for i in range(n_bins):
        lower, upper = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (confidences > lower) & (confidences <= upper)
        if mask.sum() > 0:
            bin_acc = accuracies[mask].mean()
            bin_conf = confidences[mask].mean()
            ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
    return float(ece)


def plot_confusion_matrix(cm: np.ndarray, class_names, path: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0 if cm.size else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black"
            )

    fig.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_reliability_diagram(probs: np.ndarray, labels: np.ndarray, path: str, n_bins: int = 15) -> None:
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == labels).astype(np.float32)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    bin_accs = []
    bin_confs = []

    for i in range(n_bins):
        lower, upper = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (confidences > lower) & (confidences <= upper)
        if mask.sum() > 0:
            bin_accs.append(float(accuracies[mask].mean()))
            bin_confs.append(float(confidences[mask].mean()))
        else:
            bin_accs.append(0.0)
            bin_confs.append((lower + upper) / 2)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
    ax.bar(bin_confs, bin_accs, width=1 / n_bins * 0.9, alpha=0.8)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title("Reliability Diagram")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    plt.savefig(path, bbox_inches="tight")
    plt.close(fig)