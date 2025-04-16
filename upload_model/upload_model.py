from huggingface_hub import upload_folder
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.config import (
    REPO_ID,
    LANGUAGE,
    TASK,  
    TEST_AUDIO_FILES,
    SAMPLING_RATE,
    OUTPUT_DIR 
)

# Ensure the folder to be uploaded exists
if not os.path.exists(OUTPUT_DIR):
    raise ValueError(f"The specified model path {OUTPUT_DIR} does not exist!")

# Upload the model folder to Hugging Face
upload_folder(
    folder_path=OUTPUT_DIR,
    repo_id=REPO_ID  # The Hugging Face repository name
)

print(f"Model uploaded successfully to: https://huggingface.co/{REPO_ID}")
