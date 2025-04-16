from datasets import load_dataset, load_from_disk, Audio, DatasetDict
from transformers import WhisperTokenizer, WhisperProcessor
import os
import logging
from config.config import (
    TRAIN_DATASET_PATH,
    TEST_DATASET_PATH,
    MODEL_NAME,
    LANGUAGE,
    TASK,
    SAMPLING_RATE,
    DATA_DIR
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_loading.log'),
        logging.StreamHandler()
    ]
)

def ensure_directory_exists(path):
    """Ensure the directory exists, create if it doesn't"""
    os.makedirs(path, exist_ok=True)

def load_and_prepare_datasets():
    # Ensure data directory exists
    ensure_directory_exists(DATA_DIR)
    
    # Check if datasets exist locally
    if not os.path.exists(TRAIN_DATASET_PATH) or not os.path.exists(TEST_DATASET_PATH):
        logging.info("Downloading and preparing datasets...")
        # Load the dataset from Hugging Face
        medibeng = DatasetDict()
        
        # Load train and test splits
        medibeng["train"] = load_dataset("pr0mila-gh0sh/MediBeng", split="train")
        medibeng["test"] = load_dataset("pr0mila-gh0sh/MediBeng", split="test")
        
        # Save the datasets
        medibeng["train"].save_to_disk(TRAIN_DATASET_PATH)
        medibeng["test"].save_to_disk(TEST_DATASET_PATH)
        logging.info(f"Datasets saved to {DATA_DIR}")
    
    # Load the datasets from disk
    logging.info("Loading datasets from disk...")
    train_dataset = load_from_disk(TRAIN_DATASET_PATH)
    test_dataset = load_from_disk(TEST_DATASET_PATH)

    # Log original dataset sizes
    logging.info(f"Original training data size: {len(train_dataset)}")
    logging.info(f"Original test data size: {len(test_dataset)}")

    # Take 20% of the data for both training and testing with proper shuffling
    train_size = int(0.2 * len(train_dataset))
    test_size = int(0.2 * len(test_dataset))
    
    # Shuffle and select with a fixed seed for reproducibility
    train_dataset = train_dataset.shuffle(seed=42).select(range(train_size))
    test_dataset = test_dataset.shuffle(seed=42).select(range(test_size))
    
    logging.info(f"Selected training data size: {len(train_dataset)}")
    logging.info(f"Selected test data size: {len(test_dataset)}")

    # Initialize the Whisper tokenizer and processor
    logging.info("Initializing tokenizer and processor...")
    tokenizer = WhisperTokenizer.from_pretrained(MODEL_NAME, language=LANGUAGE, task=TASK)
    processor = WhisperProcessor.from_pretrained(MODEL_NAME, language=LANGUAGE, task=TASK)

    # Cast audio column to the required format with a sampling rate of 16000 Hz
    logging.info("Preprocessing audio data...")
    train_dataset = train_dataset.cast_column("audio", Audio(sampling_rate=SAMPLING_RATE))
    test_dataset = test_dataset.cast_column("audio", Audio(sampling_rate=SAMPLING_RATE))

    # Define a function to prepare the dataset
    def prepare_dataset(batch):
        # Load and resample audio data from 48 to 16kHz
        audio = batch["audio"]
        
        # Compute log-Mel input features from the input audio array
        batch["input_features"] = processor.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
        
        # Encode target text to label ids
        batch["labels"] = tokenizer(batch["translation"]).input_ids
        return batch

    # Apply the prepare_dataset function to both train and test datasets
    logging.info("Preparing datasets for training...")
    train_dataset = train_dataset.map(prepare_dataset, remove_columns=["text","speaker_name","utterance_pitch_mean","utterance_pitch_std"])
    test_dataset = test_dataset.map(prepare_dataset, remove_columns=["text","speaker_name","utterance_pitch_mean","utterance_pitch_std"])

    logging.info("Dataset preparation completed successfully")
    return train_dataset, test_dataset, processor, tokenizer
