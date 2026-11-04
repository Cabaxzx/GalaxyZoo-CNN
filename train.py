
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from torch.amp import autocast


class FullHierarchicalLoss(nn.Module):
    """
    Custom loss function for flat regression.
    Merges loss validation with global structural Mean Squared Error penalties.
    """
    def __init__(self, lambda_reg=0.25):
        super().__init__()
        #standard loss, can be MSE or BCE: depending on the loss the architecture and evaluation must be changed
        # (apply sigmoid in model outputs or in evaluation)
        self.base_loss = nn.BCEWithLogitsLoss()
        # lambda_reg controls the amplitude of the structural penalty assigned to probability violations.
        self.lambda_reg = lambda_reg

    def forward(self, logits, targets):
        l = self.base_loss(logits, targets)

        # De-soften values to permit absolute mathematical comparisons.
        #If using MSE
        #probs = logits
        #If using BCE
        probs = torch.sigmoid(logits)
        real_probs = (probs - 0.01) / 0.98

        q1_sum = torch.sum(real_probs[:, 0:3], dim=1)
        q1_1_smooth = real_probs[:, 0]
        q1_2_disk = real_probs[:, 1]

        q2_sum = torch.sum(real_probs[:, 3:5], dim=1)
        q7_sum = torch.sum(real_probs[:, 5:8], dim=1)
        q6_sum = torch.sum(real_probs[:, 8:10], dim=1)
        q6_1_odd = real_probs[:, 8]
        q8_sum = torch.sum(real_probs[:, 10:17], dim=1)

        # Structural constraints computation.
        # MSE is applied to quantify the degree to which a child violates its parent probability distribution.
        v_q1 = F.mse_loss(q1_sum, torch.ones_like(q1_sum))
        v_q6 = F.mse_loss(q6_sum, torch.ones_like(q6_sum))
        v_q2 = F.mse_loss(q2_sum, q1_2_disk)
        v_q7 = F.mse_loss(q7_sum, q1_1_smooth)
        v_q8 = F.mse_loss(q8_sum, q6_1_odd)

        l_constraint = v_q1 + v_q2 + v_q7 + v_q6 + v_q8
        return l + (self.lambda_reg * l_constraint)


def perform_train_loop(model, loss_fn, optimizer, train_loader, device, scaler, architecture, scheduler, task="classification"):
    """Executes a single epoch iteration across the training dataloader."""
    total_loss, correct, total = 0.0, 0, 0
    model.train()

    for X, y in train_loader:
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        # Autocast initiates mixed-precision context, minimizing VRAM utilization and executing calculations in float16.
        with autocast(device_type='cuda', dtype=torch.float16):
            pred = model(X)
            if architecture == "binary":
                loss = loss_fn(pred, y.float().unsqueeze(1))
            else:
                loss = loss_fn(pred, y)

        # Scaler operations amplify gradients to circumvent float16 rounding constraints.
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if scheduler and not isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
            scheduler.step()

        total_loss += loss.item()

        with torch.no_grad():
            if architecture == "classification":
                correct += (pred.argmax(dim=1) == y).sum().item()
                total += y.size(0)
            elif architecture == "binary":
                correct += ((torch.sigmoid(pred) > 0.5).long() == y.unsqueeze(1)).sum().item()
                total += y.size(0)

    return total_loss, (correct / total if total > 0 else 0.0)


def perform_validation_loop(model, loss_fn, validation_loader, device, architecture, task="classification"):
    """Executes evaluation over the validation dataset."""
    total_loss, correct, total = 0.0, 0, 0
    model.eval()

    with torch.no_grad():
        for X, y in validation_loader:
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)

            with autocast(device_type='cuda', dtype=torch.float16):
                pred = model(X)

                if architecture == "binary":
                    loss = loss_fn(pred, y.float().unsqueeze(1))
                else:
                    loss = loss_fn(pred, y)

            total_loss += loss.item()

            if architecture == "classification":
                correct += (pred.argmax(dim=1) == y).sum().item()
                total += y.size(0)
            elif architecture == "binary":
                correct += ((torch.sigmoid(pred) > 0.5).long() == y.unsqueeze(1)).sum().item()
                total += y.size(0)

    return total_loss, (correct / total if total > 0 else 0.0)


def fit(epochs, loss_function, model, optimizer, train_loader, validation_loader, device, scaler,
        architecture="classification", scheduler=None, patience=7):
    """
    Main training function.
    Monitors epoch-level metrics.
    """
    train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []
    best_val_loss = float('inf')
    epochs_no_improve = 0
    task_name = "tree" if architecture in ["tree", "flat"] else "classification"

    for epoch in range(epochs):
        train_loss, train_acc = perform_train_loop(model, loss_function, optimizer, train_loader, device, scaler, architecture, scheduler, task=task_name)
        val_loss, val_acc = perform_validation_loop(model, loss_function, validation_loader, device, architecture, task=task_name)

        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(validation_loader)

        train_losses.append(avg_train)
        val_losses.append(avg_val)

        #Include accuracies for classification tasks
        if architecture in ["classification", "binary"]:
            train_accuracies.append(train_acc)
            val_accuracies.append(val_acc)

        if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
            scheduler.step(avg_val)

        current_lr = optimizer.param_groups[0]['lr']

        if architecture in ["classification", "binary"]:
            print(f"Epoch {epoch + 1}/{epochs} | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f} | Train Acc: {train_acc * 100:.2f}% | Val Acc: {val_acc * 100:.2f}% | LR: {current_lr:.6f}")
        else:
            print(f"Epoch {epoch + 1}/{epochs} | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f} | LR: {current_lr:.6f}")

        # Validation evaluation block triggering system checkpointing or early termination.
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            epochs_no_improve = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience and isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
                print("Early stopping sequence initiated. Model optimization convergence achieved.")
                break

    return train_losses, val_losses, train_accuracies, val_accuracies