import os
import pandas as pd
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.config import (
    REPO_ID,
    LANGUAGE,
    TASK,  # Ensure TASK is being imported as a string
    TEST_AUDIO_FILES,
    SAMPLING_RATE
)

# Set model path and language/task
model_path = REPO_ID
# Load model and processor from the specified path
processor = WhisperProcessor.from_pretrained(model_path)
model = WhisperForConditionalGeneration.from_pretrained(model_path)

# Get forced decoder IDs for translation task to English
forced_decoder_ids = processor.get_decoder_prompt_ids(language=LANGUAGE, task=TASK)

# List of audio files to process
audio_files = TEST_AUDIO_FILES

# List to store results
results = []

# Process each audio file
for audio_file in audio_files:
    # Load and preprocess the audio file using librosa
    audio_input, _ = librosa.load(audio_file, sr=SAMPLING_RATE)

    # Process the audio sample into input features for the Whisper model
    input_features = processor(audio_input, sampling_rate=SAMPLING_RATE, return_tensors="pt").input_features

    # Generate token ids for transcription/translation
    predicted_ids = model.generate(input_features, forced_decoder_ids=forced_decoder_ids)

    # Decode token ids to text (translation)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

    # Extract the file name without the extension
    file_name = os.path.basename(audio_file).replace(".wav", "")

    # Append the result to the list
    results.append([file_name, transcription])

# Create a DataFrame with the results
df = pd.DataFrame(results, columns=["Audio Name", "Translation"])

# Display the DataFrame
print(df)

# Save the results to a CSV file
output_csv_path = "audio_list_translation_results.csv"
df.to_csv(output_csv_path, index=False)

# Display the DataFrame
