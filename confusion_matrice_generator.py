import matplotlib.pyplot as plt
import numpy as np

def plot_confusion_matrix(cm, class_names):
    """
    Plot a confusion matrix with:
    - Predicted labels on top
    - Accuracy printed at the bottom
    - Colorbar scaled from 0 to 1
    """
    cm = np.array(cm, dtype=float)
    cm_normalized = cm / cm.sum(axis=1, keepdims=True)  # Normalize row-wise

    accuracy = np.trace(cm) / np.sum(cm)

    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(cm_normalized,cmap="Blues" ,vmin=0, vmax=1)

    # Ticks & labels
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)

    # Axis labels
    ax.set_xlabel("Predicted label", fontsize=12, fontweight="bold", labelpad=15)
    ax.set_ylabel("True label", fontsize=12, fontweight="bold")



    # Put predicted labels on top
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()

    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=0, rotation_mode="anchor")

    # Annotate each cell with count & probability
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, f"{cm[i, j]:.0f}\n({cm_normalized[i, j]:.2f})",
                    ha="center", va="center",
                    color="white" if cm_normalized[i, j] > 0.5 else "black")

    # Accuracy at the bottom
    fig.text(0.53, 0.05, f"Accuracy: {accuracy*100:.2f}%", ha='center', fontweight="bold", fontsize=12)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    plt.savefig('c_cm.png')

# Example usage
cm_example = np.array([[77, 23],
                       [45, 55]])
plot_confusion_matrix(cm_example, ["HCC827", "A549"])
