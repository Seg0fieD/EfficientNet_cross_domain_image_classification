# dataset.py

import os
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, ConcatDataset
from config import Config

def load_pacs_dataset(domain):
    """
    Load images from a specific domain in the PACS dataset.
    Args:
        domain (str): Name of the domain (e.g., "art_painting").
    Returns:
        dataset (Dataset): Loaded dataset.
    """
    transform = transforms.Compose([
        transforms.Resize(Config.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet normalization
    ])

    data_path = os.path.join(Config.DATA_DIR, domain)
    dataset = datasets.ImageFolder(root=data_path, transform=transform)
    return dataset

def get_dataloaders(source_domains, target_domain, batch_size=Config.BATCH_SIZE):
    """
    Create dataloaders for source (training) and target (testing) domains.
    Args:
        source_domains (list): List of domains to use for training (e.g., ["art_painting", "cartoon", "photo"]).
        target_domain (str): Domain to use for testing (e.g., "sketch").
    Returns:
        train_loader (DataLoader): Dataloader for the source domains.
        test_loader (DataLoader): Dataloader for the target domain.
    """
    # Load and combine datasets from all source domains
    train_datasets = []
    for domain in source_domains:
        dataset = load_pacs_dataset(domain)
        train_datasets.append(dataset)
    train_dataset = ConcatDataset(train_datasets) if train_datasets else None  # Handle empty list

    # Load target domain dataset
    test_dataset = load_pacs_dataset(target_domain)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) if train_dataset else None
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader