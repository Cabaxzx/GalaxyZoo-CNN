import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm
from torch.amp import autocast
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, r2_score, f1_score, accuracy_score
from sklearn.preprocessing import label_binarize

def visualize_dataset_distributions(labels_path="data/labels.csv"):
    """
    Visualizes the galaxyzoo dataset: the counts of galaxies in each category according to question 1 and the
    vote fractions for the questions that the model will be trained on for the regression tasks.
    """

    print("Loading dataset...")
    df = pd.read_csv(labels_path)

    class_probs = df[['Class1.1', 'Class1.2', 'Class1.3']].values
    hard_labels = class_probs.argmax(axis=1)

    label_map = {0: 'Smooth', 1: 'Disk', 2: 'Artifact'}
    mapped_labels = [label_map[l] for l in hard_labels]

    fig1 = plt.figure(figsize=(10, 6))

    sns.countplot(x=mapped_labels, hue=mapped_labels, order=['Smooth', 'Disk', 'Artifact'],
                  palette='viridis', edgecolor='black', legend=False)

    plt.title('Classification Task: Discrete Class Distribution', fontsize=18, fontweight='bold')
    plt.xlabel('Galaxy Morphology Type', fontsize=14)
    plt.ylabel('Number of Images', fontsize=14)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.show()

    regression_cols = [
        'Class1.1', 'Class1.2', 'Class1.3',
        'Class2.1', 'Class2.2',
        'Class7.1', 'Class7.2', 'Class7.3',
        'Class6.1', 'Class6.2',
        'Class8.1', 'Class8.2', 'Class8.3', 'Class8.4', 'Class8.5', 'Class8.6', 'Class8.7'
    ]

    col_names = [
        '1.1 (Smooth)', '1.2 (Disk)', '1.3 (Artifact)',
        '2.1 (Edge-On)', '2.2 (Face-On)',
        '7.1 (Round)', '7.2 (In-between)', '7.3 (Cigar)',
        '6.1 (Odd: Yes)', '6.2 (Odd: No)',
        '8.1 (Ring)', '8.2 (Lens)', '8.3 (Disturbed)',
        '8.4 (Irregular)', '8.5 (Other)', '8.6 (Merger)', '8.7 (Dust Lane)'
    ]

    ncols = 4
    nrows = int(np.ceil(len(regression_cols) / ncols))

    fig2, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(22, 4.5 * nrows))
    fig2.suptitle('Regression Task: Vote Fractions Distribution', fontsize=26, fontweight='bold')
    axes = axes.flatten()

    shared_bins = np.linspace(0, 1, 50)

    for i, (col, name) in enumerate(zip(regression_cols, col_names)):
        sns.histplot(df[col], bins=shared_bins, ax=axes[i], color='teal', alpha=0.6, edgecolor='black')
        axes[i].set_title(name, fontsize=16)
        axes[i].set_xlim(-0.05, 1.05)
        axes[i].set_xlabel('Vote Fraction', fontsize=12)
        axes[i].set_ylabel('Frequency', fontsize=12)
        axes[i].grid(alpha=0.2)

    for j in range(len(regression_cols), len(axes)):
        axes[j].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.show()



def plot_confusion_matrix(true_labels, predicted_labels, class_names):
    """
    Generates a normalized confusion matrix.
    Normalization standardizes rows to 1, accurately reflecting model precision regardless of class imbalance.
    """
    cm = confusion_matrix(true_labels, predicted_labels, normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(cmap='Blues', colorbar=False, ax=ax)
    ax.grid(False)
    plt.title('Test Dataset Confusion Matrix', fontsize=15)
    plt.show()


def plot_roc(true_labels, predicted_probs, class_names):
    """
    Constructs Receiver Operating Characteristic (ROC) curves.
    Evaluates classification thresholds by plotting True Positive Rate against False Positive Rate.
    """
    n_classes = len(class_names)
    plt.figure(figsize=(8, 8))

    if n_classes == 2:
        probs = predicted_probs.flatten() if predicted_probs.ndim > 1 else predicted_probs
        false_rate, true_rate, _ = roc_curve(true_labels, probs)
        roc_auc = auc(false_rate, true_rate)
        plt.plot(false_rate, true_rate, lw=2, color='darkorange', label=f'ROC {class_names[1]} (area = {roc_auc:.3f})')
    else:
        binarized_labels = label_binarize(true_labels, classes=range(n_classes))
        for i in range(n_classes):
            false_rate, true_rate, _ = roc_curve(binarized_labels[:, i], predicted_probs[:, i])
            roc_auc = auc(false_rate, true_rate)
            plt.plot(false_rate, true_rate, lw=2, label=f'ROC {class_names[i]} (area = {roc_auc:.3f})')

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0]);
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate');
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic', fontsize=15)
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.show()


def find_optimal_thresholds(val_targets, val_probs, architecture):
    """
    Derives optimal prediction boundaries to maximize the Macro F1-Score for the classification task.
    Protects imbalanced classes by shifting decision thresholds away from argmax.
    """
    print("\nStarting Threshold Optimization...")

    if architecture == "binary":
        best_t, best_f1 = 0.5, 0.0
        thresholds = np.linspace(0.01, 0.99, 99)
        for t in thresholds:
            preds = (val_probs >= t).astype(int)
            f1 = f1_score(val_targets, preds, average='macro')
            if f1 > best_f1: best_f1, best_t = f1, t
        print(f"Optimal Binary Threshold: {best_t:.3f} (Macro F1: {best_f1:.4f})")
        return best_t

    elif architecture == "classification":
        grid = np.linspace(0.1, 1.0, 10)
        t0, t1, t2 = np.meshgrid(grid, grid, grid)
        threshold_combinations = np.vstack([t0.ravel(), t1.ravel(), t2.ravel()]).T

        best_thresh, best_f1 = [1.0, 1.0, 1.0], 0.0

        for thresh_array in threshold_combinations:
            adjusted_probs = val_probs / thresh_array
            preds = np.argmax(adjusted_probs, axis=1)
            f1 = f1_score(val_targets, preds, average='macro')
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh_array

        print(f"Optimal Multi-Class Thresholds [Smooth, Disk, Artifact]: {best_thresh}")
        normalized_thresh = list(np.array(best_thresh) / max(best_thresh))
        return normalized_thresh

    return None


def evaluate_model(model, test_loader, device, class_names=None, use_tta=False, optimal_thresholds=None):
    """
    Universal test-set inference controller.
    Detects underlying architecture and outputs the corresponding evaluation plots.
    Applies optional Test Time Augmentation (TTA) via spatial inversions to stabilize predictive variance.
    """
    tta_status = "ON" if use_tta else "OFF"
    print(f"\nRunning inference on Test Set (Architecture: {model.architecture} | TTA: {tta_status})...")
    model.eval()

    all_preds, all_targets, all_probs = [], [], []

    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)

            #Constructs inference stack mapping original tensor against symmetrically flipped augmentations
            views = [X, torch.flip(X, dims=[3]), torch.flip(X, dims=[2])] if use_tta else [X]

            with autocast(device_type='cuda', dtype=torch.float16):
                batch_probs = []
                for view in views:
                    raw_outputs = model(view)
                    if model.architecture == "classification":
                        probs = torch.nn.functional.softmax(raw_outputs, dim=1)
                    elif model.architecture == "binary":
                        probs = torch.sigmoid(raw_outputs).squeeze()
                    elif model.architecture == "flat":
                        #If using BCE
                        probs = torch.sigmoid(raw_outputs)
                        #If using MSE
                        #probs = raw_outputs
                        probs = torch.where(probs < 0.04, torch.tensor(0.01, device=probs.device), probs)
                        probs = torch.clamp((probs - 0.01) / 0.98, min=0.0, max=1.0)
                    elif model.architecture == "tree":
                        probs = raw_outputs

                    batch_probs.append(probs)

                avg_probs = sum(batch_probs) / len(batch_probs)

                # Architechture specific routing
                if model.architecture == "classification":
                    if optimal_thresholds is not None:
                        thresh_tensor = torch.tensor(optimal_thresholds, device=device) + 1e-8
                        preds = torch.argmax(avg_probs / thresh_tensor, dim=1)
                    else:
                        preds = torch.argmax(avg_probs, dim=1)
                    all_probs.extend(avg_probs.cpu().numpy())

                elif model.architecture == "binary":
                    thresh = optimal_thresholds if optimal_thresholds is not None else 0.5
                    preds = (avg_probs >= thresh).long()
                    all_probs.extend(avg_probs.cpu().numpy())

                elif model.architecture in ["flat", "tree"]:
                    preds = avg_probs

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y.cpu().numpy())

    all_preds, all_targets = np.array(all_preds), np.array(all_targets)

    #Classification metrics
    if model.architecture in ["classification", "binary"]:
        all_probs = np.array(all_probs)
        class_names = class_names or (['Smooth', 'Disk'] if model.architecture == "binary" else ['Smooth', 'Disk', 'Artifact'])
        plot_confusion_matrix(all_targets, all_preds, class_names)
        plot_roc(all_targets, all_probs, class_names)

    #Regression metrics
    elif model.architecture in ["flat", "tree"]:
        target_cols = [
            'Class1.1 (Smooth)', 'Class1.2 (Disk)', 'Class1.3 (Artifact)',
            'Class2.1 (Edge-On)', 'Class2.2 (Face-On)',
            'Class7.1 (Round)', 'Class7.2 (In-between)', 'Class7.3 (Cigar)',
            'Class6.1 (Odd: Yes)', 'Class6.2 (Odd: No)',
            'Class8.1 (Ring)', 'Class8.2 (Lens)', 'Class8.3 (Disturbed)',
            'Class8.4 (Irregular)', 'Class8.5 (Other)', 'Class8.6 (Merger)', 'Class8.7 (Dust Lane)'

        ]

        shared_bins = np.linspace(-0.05, 1.05, 40)
        ncols = 4
        nrows = int(np.ceil(len(target_cols) / ncols))

        #1D Overlap Histograms
        fig1, axes1 = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 4.5 * nrows))
        fig1.suptitle(f'1D Distributions ({model.architecture.capitalize()})', fontsize=26, fontweight='bold')
        axes1 = axes1.flatten()

        for i, col in enumerate(target_cols):
            r2 = r2_score(all_targets[:, i], all_preds[:, i])
            sns.histplot(all_targets[:, i], bins=shared_bins, ax=axes1[i], label='True', color='blue', alpha=0.4)
            sns.histplot(all_preds[:, i], bins=shared_bins, ax=axes1[i], label='Pred', color='orange', alpha=0.4)
            axes1[i].set_title(f'{col} (R² = {r2:.3f})', fontsize=14)
            axes1[i].legend()

        #Hide empty subplots if the targets don't perfectly fill the grid
        for j in range(len(target_cols), len(axes1)):
            axes1[j].axis('off')

        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        plt.show()

        #2D Density Heatmaps
        fig2, axes2 = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 4.5 * nrows))
        fig2.suptitle(f'2D Density Heatmaps ({model.architecture.capitalize()})', fontsize=26, fontweight='bold')
        axes2 = axes2.flatten()

        for i, col in enumerate(target_cols):
            h = axes2[i].hist2d(x=all_targets[:, i], y=all_preds[:, i], bins=50, range=[[0, 1], [0, 1]],
                                cmap='plasma', cmin=1, norm=LogNorm())
            axes2[i].plot([0, 1], [0, 1], transform=axes2[i].transAxes, color='cyan', linestyle='--', linewidth=1.5)
            axes2[i].set_title(f'{col}', fontsize=14)
            fig2.colorbar(h[3], ax=axes2[i])

        for j in range(len(target_cols), len(axes2)):
            axes2[j].axis('off')

        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        plt.show()


def get_raw_probabilities(model, dataloader, device):
    """
    Silent extraction function compiling probability tensors.
    Required as prerequisite data structure for boundary optimization routines.
    """
    model.eval()
    all_targets, all_probs = [], []
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with autocast(device_type='cuda', dtype=torch.float16):
                raw_outputs = model(X)
                if model.architecture == 'classification':
                    probs = torch.nn.functional.softmax(raw_outputs, dim=1)
                elif model.architecture == 'binary':
                    probs = torch.sigmoid(raw_outputs).squeeze()
                elif model.architecture in ['flat', 'tree']:
                    probs = raw_outputs
            all_probs.extend(probs.cpu().numpy())
            all_targets.extend(y.cpu().numpy())
    return np.array(all_targets), np.array(all_probs)


def execute_final_evaluation(model, val_loader, test_loader, device, class_names=None):
    """
    Automated sequential execution controller combining probability extraction, threshold optimization, and spatial metrics routing.
    """
    print('INITIATING FINAL EVALUATION ON TEST DATASET')
    val_targets, val_probs = get_raw_probabilities(model, val_loader, device)

    if model.architecture in ["classification", "binary"]:
        print("\nOptimizing decision thresholds...")
        optimal_thresh = find_optimal_thresholds(val_targets, val_probs, model.architecture)
    else:
        optimal_thresh = None

    print("nExecuting evaluation on Test Set with TTA...")
    evaluate_model(model=model, test_loader=test_loader, device=device, class_names=class_names, use_tta=True,
                   optimal_thresholds=optimal_thresh)


def plot_loss(train_losses, val_losses, train_accuracies=None, val_accuracies=None, task="classification"):
    """
    Compiles standard convergence tracking plots validating learning progression and detecting epoch overfit limits.
    """
    epochs = range(1, len(train_losses) + 1)
    if task == 'classification':
        fig, axs = plt.subplots(2, figsize=(10, 8))
        axs[0].plot(epochs, train_losses, label='train loss', color='blue', linewidth=2)
        axs[0].plot(epochs, val_losses, label='validation loss', color='orange', linewidth=2)
        axs[0].set_title('Classification Loss', fontsize=15);
        axs[0].set_xlabel('Epoch');
        axs[0].set_ylabel("Cross Entropy");
        axs[0].legend();
        axs[0].grid(alpha=0.3)
        if train_accuracies and val_accuracies:
            axs[1].plot(epochs, train_accuracies, label='train accuracy', color='green', linewidth=2)
            axs[1].plot(epochs, val_accuracies, label='validation accuracy', color='purple', linewidth=2)
            axs[1].set_title('Classification Accuracy', fontsize=15);
            axs[1].set_xlabel('Epoch');
            axs[1].set_ylabel('Accuracy');
            axs[1].legend();
            axs[1].grid(alpha=0.3)
        plt.tight_layout();
        plt.show()
    else:
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, train_losses, label='train loss', color='blue', linewidth=2)
        plt.plot(epochs, val_losses, label='validation loss', color='orange', linewidth=2)
        plt.title(f'Loss ({task.capitalize()})', fontsize=15);
        plt.xlabel('Epoch');
        plt.ylabel('Loss')
        plt.legend();
        plt.grid(alpha=0.3);
        plt.tight_layout();
        plt.show()