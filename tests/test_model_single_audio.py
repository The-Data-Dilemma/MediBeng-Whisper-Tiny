# from transformers import WhisperProcessor, WhisperForConditionalGeneration
import os
import pandas as pd
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.config import (
    REPO_ID,
    LANGUAGE,
    TASK,  
    TEST_AUDIO_FILES,
    SAMPLING_RATE
)

# Load model and processor from the specified path
processor = WhisperProcessor.from_pretrained(REPO_ID)
model = WhisperForConditionalGeneration.from_pretrained(REPO_ID)

# Get forced decoder IDs for translation task to English
forced_decoder_ids = processor.get_decoder_prompt_ids(language=LANGUAGE, task=TASK)

# Path to your single audio file
audio_file_path = TEST_AUDIO_FILES[0]

# Load and preprocess the audio file using librosa
audio_input, _ = librosa.load(audio_file_path, sr=SAMPLING_RATE)

# Process the audio sample into input features for the Whisper model
input_features = processor(audio_input, sampling_rate=SAMPLING_RATE, return_tensors="pt").input_features

# Generate token ids for the transcription/translation
predicted_ids = model.generate(input_features, forced_decoder_ids=forced_decoder_ids)

# Decode token ids to text (translation)
translation = processor.batch_decode(predicted_ids, skip_special_tokens=True)

# Output the transcription/translation result
print("Translation:", translation[0])