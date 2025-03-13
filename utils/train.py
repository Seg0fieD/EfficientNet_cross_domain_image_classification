import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from config import Config
from models.efficientnet import EfficientNet
from utils.dataset import get_dataloaders
from utils.logger import Logger
from utils.evaluate import evaluate_model  # Import the updated evaluate_model

def train_model(source_domains, target_domain):
    """
    Train the EfficientNet model on multiple source domains and evaluate on the target domain.
    Args:
        source_domains (list): List of domains to use for training (e.g., ["art_painting", "cartoon", "photo"]).
        target_domain (str): Domain to use for testing (e.g., "sketch").
    """
    # Set device
     # Set device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    # Initialize logger
    logger = Logger()

    # Load data
    train_loader, test_loader = get_dataloaders(source_domains, target_domain)

    # Initialize model, loss, and optimizer
    model = EfficientNet(version="b0", num_classes=len(Config.CLASSES)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    # Gradient accumulation steps
    accumulation_steps = Config.ACCUMULATION_STEPS
    optimizer.zero_grad()  # Initialize gradients

    # Training loop
    for epoch in range(Config.NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        for i, (images, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{Config.NUM_EPOCHS}")):
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss = loss / accumulation_steps  # Normalize loss for accumulation
            running_loss += loss.item()

            # Backward pass
            loss.backward()

            # Update weights every `accumulation_steps` batches
            if (i + 1) % accumulation_steps == 0:
                optimizer.step()  # Update weights
                optimizer.zero_grad()  # Reset gradients

        # Show average loss for the epoch
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}\n ")

        # Calculate average loss for the epoch
        epoch_loss = running_loss / len(train_loader)

        # Evaluate the model on the target domain
        accuracy = evaluate_model(model, test_loader, device, target_domain = target_domain)  

        # Log metrics
        logger.log_metrics(epoch, epoch_loss, accuracy)

    # Save the model
    os.makedirs(Config.SAVE_DIR, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(Config.SAVE_DIR, Config.MODEL_NAME))
    print(f"Model saved to {Config.SAVE_DIR}/{Config.MODEL_NAME}")

    # Save plots
    logger.save_plots()



# # Show average loss for the epoch
#         print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}\n~ • ~ • ~ • ~ • ~ • ~ • ~ • ~ • ~ • ~ • ~ • ~ \n")


   