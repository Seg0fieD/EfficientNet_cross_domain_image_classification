# main.py

from utils.train import train_model
from utils.evaluate import evaluate_model
from utils.dataset import get_dataloaders
from models.efficientnet import EfficientNet
from config import Config
import torch
import os

if __name__ == "__main__":
    source_domains = ["art_painting", "cartoon", "photo"]  # Train 
    target_domain = "sketch"  # Evaluate 

    # Model Training
    train_model(source_domains, target_domain)

    # Evaluate the model
    # Set device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    # Load the trained model
    model = EfficientNet(version="b0", num_classes=len(Config.CLASSES)).to(device)
    model.load_state_dict(torch.load(os.path.join(Config.SAVE_DIR, Config.MODEL_NAME)))

    # Load the test dataloader for the target domain
    _, test_loader = get_dataloaders([], target_domain) 

    # Evaluate the model
    evaluate_model(model, test_loader, device, target_domain)