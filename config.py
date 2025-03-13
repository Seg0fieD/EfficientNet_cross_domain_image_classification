class Config:
    # path of the dataset
    DATA_DIR = "data/PACS"
    DOMAINS = ["art_painting", "cartoon", "photo", "sketch"]
    CLASSES = ["dog", "elephant", "giraffe","guitar", "horse", "house", "person"]

    # Training hyperparamenters
    BATCH_SIZE = 32
    ACCUMULATION_STEPS = 4
    NUM_EPOCHS = 100
    LEARNING_RATE = 0.001
    IMAGE_SIZE = (224, 224)  # image input size

    # Model saving
    SAVE_DIR = "saved_models"
    MODEL_NAME = "efficientNet_PACS.pth"

    