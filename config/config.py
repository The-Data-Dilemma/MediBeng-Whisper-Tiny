import os

# Dataset paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = "data"
DATASET= "pr0mila-gh0sh/MediBeng"
TRAIN_DATASET_PATH = os.path.join(DATA_DIR, "train_dataset")
TEST_DATASET_PATH = os.path.join(DATA_DIR, "test_dataset")

# Model configuration
MODEL_NAME = "openai/whisper-tiny"
LANGUAGE = "English"
TASK = "translate"

# Training configuration
OUTPUT_DIR = "MediBeng-Whisper-Tiny"
TRAIN_BATCH_SIZE = 1
EVAL_BATCH_SIZE = 1
LEARNING_RATE = 1e-5
WARMUP_STEPS = 50
MAX_STEPS = 500
GRADIENT_ACCUMULATION_STEPS = 1
SAVE_STEPS = 50
EVAL_STEPS = 50
LOGGING_STEPS = 25
GENERATION_MAX_LENGTH = 225

# Audio configuration
SAMPLING_RATE = 16000 

# Hugging Face Repository Name
REPO_ID = "username/MediBeng" 

# Test audio files to test the model 

TEST_AUDIO_FILES = [
    "data/Female-Bengali-English-2045.wav",
    "data/Female-Bengali-English-2065.wav",
    "data/Female-Bengali-English-2072.wav",
    "data/Male-Bengali-English-1959.wav",
    "data/Male-Bengali-English-2372.wav",
    "data/Male-Bengali-English-2338.wav"
]
