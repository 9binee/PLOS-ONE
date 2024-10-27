import torch
import os
import copy
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from torch.nn.functional import softmax

def cnn_train(model, train_loader, loss_func, optimizer):
    """
    Train the CNN model for one epoch.
    
    Parameters:
    - model: The CNN model to be trained.
    - train_loader: DataLoader containing the training dataset.
    - loss_func: Loss function used to calculate the loss during training.
    - optimizer: Optimizer used to update the model's weights.

    Returns:
    - avg_loss: The average training loss for the epoch.
    - accuracy: The training accuracy for the epoch.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    num_batches = 0

    for images, labels, _ in train_loader:
        images = images.to(device, dtype=torch.float)
        labels = labels.to(device, dtype=torch.long)

        # Zero the parameter gradients
        optimizer.zero_grad()
        # Forward pass
        output = model(images)
        # Calculate loss
        loss = loss_func(output, labels)
        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Accumulate loss and accuracy
        running_loss += loss.item()
        _, predicted = torch.max(output, dim=1)
        total += len(labels)
        correct += (predicted == labels).sum().item()
        num_batches += 1

    avg_loss = running_loss / num_batches
    accuracy = correct / total

    return avg_loss, accuracy


def cnn_valid(model, valid_loader, loss_func):
    """
    Validate the CNN model on the validation dataset.

    Parameters:
    - model: The CNN model to be validated.
    - valid_loader: DataLoader containing the validation dataset.
    - loss_func: Loss function used to calculate the validation loss.

    Returns:
    - avg_loss: The average validation loss for the epoch.
    - acc: The validation accuracy for the epoch.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    num_batches = 0

    with torch.no_grad():
        for batch_idx, (inputs, labels, _) in enumerate(valid_loader):
            images = inputs.to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.long)

            # Forward pass
            output = model(images)
            # Calculate loss
            loss = loss_func(output, labels)

            # Accumulate loss and accuracy
            running_loss += loss.item()
            _, predicted = torch.max(output, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            num_batches += 1

    avg_loss = running_loss / num_batches
    acc = correct / total

    return avg_loss, acc


def cnn_test(model, test_loader, loss_func):
    """
    Test the CNN model on the test dataset and calculate additional classification metrics.

    Parameters:
    - model: The CNN model to be tested.
    - test_loader: DataLoader containing the test dataset.
    - loss_func: Loss function used to calculate the test loss.

    Returns:
    - avg_loss: The average test loss.
    - accuracy: The accuracy on the test dataset.
    - precision, recall, f1: Precision, recall, and F1-score for binary classification.
    - auc: Area under the ROC curve (AUC).
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    num_batches = 0

    all_labels = []
    all_predicted = []
    all_probs = []

    with torch.no_grad():
        for images, labels, _ in test_loader:
            images = images.to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.long)

            # Forward pass
            output = model(images)
            # Calculate loss
            loss = loss_func(output, labels)

            # Accumulate loss and accuracy
            running_loss += loss.item()
            _, predicted = torch.max(output, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            num_batches += 1

            # Collect all labels, predictions, and probabilities for further metrics calculation
            all_labels.extend(labels.cpu().numpy())
            all_predicted.extend(predicted.cpu().numpy())
            probs = softmax(output, dim=1)
            all_probs.extend(probs.cpu().numpy())

    avg_loss = running_loss / num_batches
    accuracy = correct / total

    # Convert labels to binary format for precision, recall, and F1-score calculation
    binary_labels = [1 if label == 2 else 0 for label in all_labels]
    binary_predicted = [1 if pred == 2 else 0 for pred in all_predicted]

    # Calculate binary classification metrics
    precision = precision_score(binary_labels, binary_predicted)
    recall = recall_score(binary_labels, binary_predicted)
    f1 = f1_score(binary_labels, binary_predicted)

    # Extract the probability of the positive class (label 2) for AUC calculation
    positive_probs = [prob[2] for prob in all_probs]
    auc = roc_auc_score(binary_labels, positive_probs)

    return avg_loss, accuracy, precision, recall, f1, auc


def cnn_learning(model, train_loader, valid_loader, loss_func, optimizer, save_directory, model_name, max_epoch=500, patience=20):
    """
    Train the CNN model with early stopping based on validation loss.

    Parameters:
    - model: The CNN model to be trained.
    - train_loader: DataLoader containing the training dataset.
    - valid_loader: DataLoader containing the validation dataset.
    - loss_func: Loss function used to calculate the training and validation losses.
    - optimizer: Optimizer used to update the model's weights.
    - save_directory: Directory where the best model will be saved.
    - model_name: Name of the saved model file.
    - max_epoch: Maximum number of training epochs. Default is 500.
    - patience: Number of epochs to wait without improvement before stopping. Default is 30.

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
        train_loss, train_acc = cnn_train(model, train_loader, loss_func, optimizer)
        val_loss, val_acc = cnn_valid(model, valid_loader, loss_func)

        if epoch % 10 == 0:
            print(f"Epoch {epoch+1}/{max_epoch}")
            print(f"Train Loss: {train_loss:.4f}  Train Accuracy: {train_acc:.4f}")
            print(f"Validation Loss: {val_loss:.4f}  Validation Accuracy: {val_acc:.4f}")

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
