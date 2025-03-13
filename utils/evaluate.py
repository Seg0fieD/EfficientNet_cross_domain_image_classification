import torch
from tqdm import tqdm

def evaluate_model(model, test_loader, device, target_domain):
    """
    Evaluate the trained model on the target domain.
    Args:
        model (nn.Module): Trained model.
        test_loader (DataLoader): Dataloader for the target domain.
        device (str): Device to use for evaluation.
    Returns:
        accuracy (float): Accuracy on the target domain.
    """
    model.eval()
    correct = 0
    total = 0

     # Set device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Accuracy on target: {target_domain} = {accuracy:.2f}% \n~ • ~ • ~ • ~ • ~ • ~ • ~ • ~ • ~ • ~ • ~ • ~ \n")
    return accuracy