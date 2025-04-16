from transformers import WhisperForConditionalGeneration, Seq2SeqTrainer, Seq2SeqTrainingArguments
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import evaluate
import numpy as np
from config.config import (
    MODEL_NAME,
    LANGUAGE,
    TASK,
    OUTPUT_DIR,
    TRAIN_BATCH_SIZE,
    EVAL_BATCH_SIZE,
    LEARNING_RATE,
    WARMUP_STEPS,
    MAX_STEPS,
    GRADIENT_ACCUMULATION_STEPS,
    SAVE_STEPS,
    EVAL_STEPS,
    LOGGING_STEPS,
    GENERATION_MAX_LENGTH
)
from data_loader import load_and_prepare_datasets
import logging
from datetime import datetime

# Set up logging
current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'training_{current_time}.log'),
        logging.StreamHandler()
    ]
)

# Load and prepare datasets
logging.info("Loading and preparing datasets...")
train_dataset, test_dataset, processor, tokenizer = load_and_prepare_datasets()

# Load the Whisper model
logging.info("Loading Whisper model...")
model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
# Set language and task in generation config
model.config.forced_decoder_ids = None
model.generation_config.update(
    language=LANGUAGE.lower(),
    task=TASK,
    forced_decoder_ids=None
)

# Define the data collator for padding
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Split inputs and labels since they have to be of different lengths and need different padding methods
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # Get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # Replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # If the beginning-of-sequence token is appended, cut it here
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        # Add attention mask to the batch
        batch["attention_mask"] = torch.ones_like(batch["input_features"][0])
        batch["labels"] = labels
        return batch

# Create the data collator
data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=model.config.decoder_start_token_id,
)

# Initialize the evaluation metric
metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # Convert predictions to the correct format
    if isinstance(pred_ids, tuple):
        pred_ids = pred_ids[0]
    
    # Convert to numpy array if needed
    if isinstance(pred_ids, list):
        pred_ids = np.array(pred_ids)
    if isinstance(label_ids, list):
        label_ids = np.array(label_ids)

    # Ensure we have the correct shape
    if len(pred_ids.shape) == 3:  # (batch_size, num_beams, sequence_length)
        pred_ids = pred_ids[:, 0, :]  # Take the first beam
    elif len(pred_ids.shape) == 2:  # (batch_size, sequence_length)
        pass
    else:
        raise ValueError(f"Unexpected prediction shape: {pred_ids.shape}")

    # Convert to list of lists for the tokenizer
    pred_ids = pred_ids.tolist()
    label_ids = label_ids.tolist()

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}

# Define training arguments
logging.info("Setting up training arguments...")
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    learning_rate=LEARNING_RATE,
    warmup_steps=WARMUP_STEPS,
    max_steps=MAX_STEPS,
    gradient_checkpointing=True,
    fp16=False,
    eval_strategy="steps",
    per_device_eval_batch_size=EVAL_BATCH_SIZE,
    save_steps=SAVE_STEPS,
    eval_steps=EVAL_STEPS,
    logging_steps=LOGGING_STEPS,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False,
    predict_with_generate=True,
    generation_max_length=GENERATION_MAX_LENGTH,
    logging_dir=f"{OUTPUT_DIR}/runs/{current_time}",  # Add timestamp to TensorBoard logs
)

# Initialize the trainer
logging.info("Initializing trainer...")
trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)

# Save processor
logging.info("Saving processor...")
processor.save_pretrained(training_args.output_dir)

# Start the training process
logging.info("Starting training...")
trainer.train()

# Save the final model and tokenizer
logging.info("Saving final model and tokenizer...")
trainer.save_model(training_args.output_dir)
processor.save_pretrained(training_args.output_dir)
tokenizer.save_pretrained(training_args.output_dir)

logging.info("Training completed!")
