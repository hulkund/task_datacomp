import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

DEVICE = "cuda"


def train_fixmatch(model, labeled_dl, unlabeled_dl, val_dl, num_classes, checkpoint_path, num_epochs=30, lr=0.01, tau=0.95, temperature=1.0, lambda_u=1.0):
    """
    Train using FixMatch pseudolabeling logic.
    Args:
        model: PyTorch model.
        labeled_dl: DataLoader for labeled data.
        unlabeled_dl: DataLoader for unlabeled data (yields (weak, strong) augmented pairs).
        val_dl: DataLoader for validation data.
        num_classes: Number of classes.
        checkpoint_path: Path to save checkpoints.
        num_epochs: Number of epochs.
        lr: Learning rate.
        tau: Confidence threshold for pseudolabels.
        temperature: Softmax temperature.
        lambda_u: Weight for unlabeled loss.
    """
    device = DEVICE
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 5
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for (labeled_batch, unlabeled_batch) in zip(labeled_dl, unlabeled_dl):
            # Labeled data
            images_x, _, labels_x, _ = labeled_batch
            images_x, labels_x = images_x.to(device), labels_x.to(device)
            # Unlabeled data (weak, strong)
            (images_u_w, images_u_s), _ = unlabeled_batch
            images_u_w = images_u_w.to(device)
            images_u_s = images_u_s.to(device)
            # Supervised loss
            logits_x = model(images_x)
            Lx = criterion(logits_x, labels_x)
            # Pseudolabels from weakly augmented
            with torch.no_grad():
                logits_u_w = model(images_u_w)
                probs = torch.softmax(logits_u_w / temperature, dim=-1)
                max_probs, targets_u = torch.max(probs, dim=-1)
                mask = (max_probs >= tau).float()
            # Unlabeled loss (only for confident pseudolabels)
            logits_u_s = model(images_u_s)
            Lu = criterion(logits_u_s, targets_u)
            Lu = (Lu * mask).mean()  # Only confident samples
            # Total loss
            loss = Lx + lambda_u * Lu
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        # Validation
        model.eval()
        val_loss = 0.0
        correct, total = 0, 0
        with torch.no_grad():
            for images, _, labels, _ in val_dl:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "patience_counter": patience_counter
            }, checkpoint_path)
        else:
            patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
        print(f"Epoch {epoch} | Train loss: {running_loss:.4f} | Val loss: {val_loss:.4f} | Val acc: {100 * correct / total:.2f}%")
    return model