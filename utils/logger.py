import os
import matplotlib.pyplot as plt

class Logger:
    def __init__(self, log_dir="logs", plot_dir="plots"):
        self.log_dir = log_dir
        self.plot_dir = plot_dir
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.plot_dir, exist_ok=True)
        self.log_file = os.path.join(self.log_dir, "training_log.txt")
        self.epoch_losses = []
        self.epoch_accuracies = []

    def log_metrics(self, epoch, loss, accuracy):
        """
        Log metrics (loss and accuracy) for each epoch.
        Args:
            epoch (int): Current epoch number.
            loss (float): Training loss for the epoch.
            accuracy (float): Evaluation accuracy for the epoch.
        """
        # Append metrics to lists
        self.epoch_losses.append(loss)
        self.epoch_accuracies.append(accuracy)

        # Write metrics to log file
        with open(self.log_file, "a") as f:
            f.write(f"Epoch {epoch + 1}: Loss = {loss:.4f}  ||  Accuracy = {accuracy:.2f}%\n")

    def save_plots(self):
        """
        Save plots for training loss and evaluation accuracy.
        """
        # Plot training loss
        plt.figure()
        plt.plot(self.epoch_losses, label="Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss Over Epochs")
        plt.legend()
        plt.savefig(os.path.join(self.plot_dir, "training_loss.png"))
        plt.close()

        # Plot evaluation accuracy
        plt.figure()
        plt.plot(self.epoch_accuracies, label="Evaluation Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (%)")
        plt.title("Evaluation Accuracy Over Epochs")
        plt.legend()
        plt.savefig(os.path.join(self.plot_dir, "evaluation_accuracy.png"))
        plt.close()