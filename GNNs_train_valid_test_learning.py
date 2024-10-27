import torch
import os
import copy
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import torch.nn.functional as F

def gnn_train(model, data, loss_func, optimizer):
    """
    Train the GNN model for one epoch.

    Parameters:
    - model: The GNN model to be trained.
    - data: Input data containing features, labels, and masks for training.
    - loss_func: Loss function used to calculate the loss during training.
    - optimizer: Optimizer used to update the model's weights.

    Returns:
    - loss: The training loss for the epoch.
    - accuracy: The training accuracy for the epoch.
    """
    model.train()
    optimizer.zero_grad()
    # Forward pass
    output = model(data)
    labels = data.y[data.train_mask].to(dtype=torch.long)
    # Calculate loss
    loss = loss_func(output[data.train_mask], labels)
    # Backward pass and optimize
    loss.backward()
    optimizer.step()

    # Calculate training accuracy
    _, predicted = torch.max(output, dim=1)
    correct = (predicted[data.train_mask] == labels).sum().item()
    accuracy = correct / data.train_mask.sum().item()

    return loss.item(), accuracy


def gnn_valid(model, data, loss_func):
    """
    Validate the GNN model on the validation dataset.

    Parameters:
    - model: The GNN model to be validated.
    - data: Input data containing features, labels, and masks for validation.
    - loss_func: Loss function used to calculate the validation loss.

    Returns:
    - avg_loss: The average validation loss.
    - acc: The validation accuracy for the epoch.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        output = model(data)
        labels = data.y[data.val_mask].to(dtype=torch.long)
        # Calculate loss
        loss = loss_func(output[data.val_mask], labels)
        running_loss += loss.item()

        # Calculate validation accuracy
        _, predicted = torch.max(output, dim=1)
        total += data.val_mask.sum().item()
        correct += (predicted[data.val_mask] == labels).sum().item()

    avg_loss = running_loss
    acc = correct / total

    return avg_loss, acc


def gnn_test(model, data, loss_func):
    """
    Test the GNN model on the test dataset and calculate additional classification metrics.

    Parameters:
    - model: The GNN model to be tested.
    - data: Input data containing features, labels, and masks for testing.
    - loss_func: Loss function used to calculate the test loss.

    Returns:
    - avg_loss: The average test loss.
    - accuracy: The test accuracy.
    - precision, recall, f1: Precision, recall, and F1-score for binary classification.
    - auc: Area under the ROC curve (AUC).
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    all_labels = []
    all_predicted = []
    all_probs = []

    with torch.no_grad():
        out = model(data)
        labels = data.y[data.test_mask].to(dtype=torch.long)
        # Calculate loss
        loss = loss_func(out[data.test_mask], labels)
        running_loss += loss.item()

        # Calculate test accuracy
        _, predicted = torch.max(out[data.test_mask], 1)
        total += data.test_mask.sum().item()
        correct += (predicted == labels).sum().item()

        # Collect labels, predictions, and probabilities for further metrics calculation
        all_labels.extend(labels.cpu().numpy())
        all_predicted.extend(predicted.cpu().numpy())
        probs = F.softmax(out[data.test_mask], dim=1)
        all_probs.extend(probs.cpu().numpy())

    avg_loss = running_loss
    accuracy = correct / total

    # Convert labels to binary format for precision, recall, and F1-score calculation
    binary_labels = [1 if label == 2 else 0 for label in all_labels]
    binary_predicted = [1 if pred == 2 else 0 for pred in all_predicted]

    # Calculate binary classification metrics
    precision = precision_score(binary_labels, binary_predicted, pos_label=1)
    recall = recall_score(binary_labels, binary_predicted, pos_label=1)
    f1 = f1_score(binary_labels, binary_predicted, pos_label=1)

    # Extract the probability of the positive class (label 2) for AUC calculation
    positive_probs = [prob[2] for prob in all_probs]
    auc = roc_auc_score(binary_labels, positive_probs)

    return avg_loss, accuracy, precision, recall, f1, auc


def gnn_learning(model, data, loss_func, optimizer, save_directory, model_name, max_epoch=500, patience=20):
    """
    Train the GNN model with early stopping based on validation loss.

    Parameters:
    - model: The GNN model to be trained.
    - data: Input data containing features, labels, and masks for training and validation.
    - loss_func: Loss function used to calculate the training and validation losses.
    - optimizer: Optimizer used to update the model's weights.
    - save_directory: Directory where the best model will be saved.
    - model_name: Name of the saved model file.
    - max_epoch: Maximum number of training epochs. Default is 500.
    - patience: Number of epochs to wait without improvement before stopping. Default is 20.

    Returns:
    - best_val_loss: The lowest validation loss achieved.
    - best_val_acc: The highest validation accuracy achieved.
    """
    best_val_loss = float("Inf")
    best_val_acc = 0
    patience_counter = 0
    best_val_epoch = 0
    early_stopping_epoch = 0
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(max_epoch):
        train_loss, train_acc = gnn_train(model, data, loss_func, optimizer)
        val_loss, val_acc = gnn_valid(model, data, loss_func)

        if epoch % 20 == 0:
            print(f"Epoch {epoch+1}/{max_epoch}")
            print(f"Train Loss: {train_loss:.4f}  Train Accuracy: {train_acc:.2f}%")
            print(f"Validation Loss: {val_loss:.4f}  Validation Accuracy: {val_acc:.2f}%")

        # Save the best model based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            best_val_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
            early_stopping_epoch = epoch

        if patience_counter >= patience:
            print("Early stopping!")
            print(f"Early stopping at epoch: {early_stopping_epoch + 1}")
            print(f"Best epoch with lowest validation loss: {best_val_epoch + 1}")
            break

    # Load the best model weights and save the model
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), os.path.join(save_directory, model_name))

    return best_val_loss, best_val_acc
