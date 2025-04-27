# app/audio_processing.py
import os
import librosa
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from config.config import REPO_ID, LANGUAGE, TASK, SAMPLING_RATE

# Load model and processor globally so it's not loaded multiple times
processor = WhisperProcessor.from_pretrained(REPO_ID)
model = WhisperForConditionalGeneration.from_pretrained(REPO_ID)

# Get forced decoder IDs for translation task to English
forced_decoder_ids = processor.get_decoder_prompt_ids(language=LANGUAGE, task=TASK)

# Function to process audio and return transcription
def process_audio(audio_file_path: str):
    try:
        # Load and preprocess the audio file using librosa
        audio_input, _ = librosa.load(audio_file_path, sr=SAMPLING_RATE)

        # Process the audio sample into input features for the Whisper model
        input_features = processor(audio_input, sampling_rate=SAMPLING_RATE, return_tensors="pt").input_features

        # Generate token ids for the transcription/translation
        predicted_ids = model.generate(input_features, forced_decoder_ids=forced_decoder_ids)

        # Decode token ids to text (translation)
        translation = processor.batch_decode(predicted_ids, skip_special_tokens=True)

        return translation[0]
    except Exception as e:
        raise Exception(f"Error processing the audio file: {str(e)}")
