
<p align="right">
  <img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License" />
  <img src="https://img.shields.io/badge/Status-Fine%20Tuned-brightgreen" alt="Model Status" />
  <img src="https://img.shields.io/badge/Base%20Model-OpenAI%20Whisper%20Tiny-blue" alt="Base Model" />
  <a href="https://huggingface.co/pr0mila-gh0sh/MediBeng-Whisper-Tiny">
    <img src="https://img.shields.io/badge/Hugging%20Face-Model-blue" alt="Hugging Face Model" />
  </a>
</p>




# Medibeng Whisper Tiny: Code-Switched Bengali-English Speech Translation for Clinical Settings

## Overview

For many **genAI solutions** in the **clinical domain**, doctor-patient transcription is a crucial task. It is especially challenging in clinical settings when the language is **code-switched**, i.e., when both **Bengali and English** are mixed in the conversation. This is common in multilingual environments, particularly in healthcare.

To solve this problem, I have developed a model that **translates** the **code-switched transcription** into one language, making it easier for further processes such as analysis, record-keeping, or integrating with other AI models.

The solution is designed to transcribe and translate **code-switched Bengali-English** conversations in clinical settings, making it easier for practitioners to process the information and use it for patient records or decision-making.

## Model Details

- Base model: openai/whisper-tiny
- Fine-tuned for: translate task (code-mixed Bengali-English → English)
- Domain: Clinical/Medical
- Language support: code-mixed Bengali-English (input), English (output)

## Dataset and Model on Hugging Face <img src="https://huggingface.co/front/assets/huggingface_logo.svg" alt="Hugging Face" width="20" height="20" />

📂 **Dataset**: Check out the **MediBeng** (30% subset) dataset used to fine-tune this model! This dataset includes synthetic code-switched clinical conversations in Bengali and English. It is designed to help train models for tasks like **speech recognition (ASR)**, **text-to-speech (TTS)**, and **machine translation**, focusing on bilingual code-switching in healthcare settings.


🔗 **Full Dataset Link**: [MediBeng Dataset](https://huggingface.co/datasets/pr0mila-gh0sh/MediBeng)



🔧 **Dataset Parquet File Creation**: Here's how I loaded the dataset to Hugging Face!  
🔗 **Repo Link for Parquet-to-HuggingFace Process**: [Parquet-to-HuggingFace Process](https://github.com/pr0mila/ParquetToHuggingFace)

You can access the fine-tuned model on Hugging Face using the link below:  
🔗 **Model Link**: [MediBeng-Whisper-Tiny](https://huggingface.co/pr0mila-gh0sh/MediBeng-Whisper-Tiny)

## Model Test Example Results

Below are some results showing the **actual translations** compared to **Medibeng Whisper Tiny** model translations.

| **Audio Name**                      | **Code-Switched Bengali-English Clinical Actual Conversations**                                                           | **Actual Translation**                                                                                              | **Whisper Tiny Translation**                                                        |  **Medibeng Whisper-Tiny Translation** 🚀                                                     |
|-------------------------------------|----------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------|
| [Female-Bengali-English-2045](https://github.com/pr0mila/MediBeng-Whisper-Tiny/raw/main/tests/data/Female-Bengali-English-2045.wav)         | আপনার শারীরিক পরীক্ষা করতে হবে, and we will schedule that shortly.।                                                      | You need a physical check-up, and we will schedule that shortly.                                                      | You can cut the hair and we will schedule that shortly.                           | You need a physical check-up, and we will schedule that shortly.                              |
| [Female-Bengali-English-2065](https://github.com/pr0mila/MediBeng-Whisper-Tiny/raw/main/tests/data/Female-Bengali-English-2065.wav)         | আপনার রক্তচাপ খুব বেশি, but আমরা monitor করতে থাকব।                                                         | Your blood pressure is very high, but we will keep monitoring.                                                        | You can also find out about the national rock club in the city of Baitamra.       | Your blood pressure is very high, but we will keep monitoring.                              |
| [Female-Bengali-English-2072](https://github.com/pr0mila/MediBeng-Whisper-Tiny/raw/main/tests/data/Female-Bengali-English-2072.wav)         |আপনার শরীরের ব্যথা অনেক, please let me know if it’s severe.                                                         | You have a lot of body pain, please let me know if it’s severe.                                                         | Please let me know if it's a way out.                                             | You have a lot of body pain, please let me know if it’s severe.                             |
| [Male-Bengali-English-1959](https://github.com/pr0mila/MediBeng-Whisper-Tiny/raw/main/tests/data/Male-Bengali-English-1959.wav)           | আপনার শরীরের তাপমাত্রা 103°F, which indicates a fever.                                                               | Your body temperature is 103°F, which indicates a fever.                                                                | You should read it, the mantra actually, the Indigree Fahrenheit which indicates a fever. | Your body temperature is 103°F, which indicates a fever.                                       |
| [Male-Bengali-English-2372](https://github.com/pr0mila/MediBeng-Whisper-Tiny/raw/main/tests/data/Male-Bengali-English-2372.wav)           | আপনার হাতের আঙুলে কিছু সমস্যা হয়েছে, Let me take a closer look.                                                        | You have some issues with your fingers, Let me take a closer look.                                                     | You were the one who was the one who was the famous man. Let me take a closer look. | You have some issues with your fingers, Let me take a closer look.                          |
| [Male-Bengali-English-2338](https://github.com/pr0mila/MediBeng-Whisper-Tiny/raw/main/tests/data/Male-Bengali-English-2338.wav)           | আপনি কি নিয়মিত ব্যায়াম করেন? It’s essential for overall health.                                   | Do you exercise regularly? It’s essential for overall health.                                                          | You need a new me to BAM KORIN, it's essential for overall health.              | Do you exercise regularly? It’s essential for overall health.                                 |

The audio files used for these examples are stored in the `tests/data` directory in the repository. For example:
- `tests/data/Female-Bengali-English-2045.wav`
## Evaluation Results
The model's performance improved as the training progressed, showing consistent reduction in **training loss** and **Word Error Rate (WER)** on the evaluation set.

| **Epoch** | **Training Loss** | **Training Grad Norm** | **Learning Rate** | **Eval Loss** | **Eval WER** |
|-----------|-------------------|------------------------|-------------------|---------------|--------------|
| 0.03      | 2.6213            | 61.56                  | 4.80E-06          | -             | -            |
| 0.07      | 1.609             | 44.09                  | 9.80E-06          | 1.13          | 107.72       |
| 0.1       | 0.7685            | 52.27                  | 9.47E-06          | -             | -            |
| 0.13      | 0.4145            | 32.27                  | 8.91E-06          | 0.37          | 47.53        |
| 0.16      | 0.3177            | 17.98                  | 8.36E-06          | -             | -            |
| 0.2       | 0.222             | 7.7                    | 7.80E-06          | 0.1           | 45.19        |
| 0.23      | 0.0915            | 1.62                   | 7.24E-06          | -             | -            |
| 0.26      | 0.081             | 0.4                    | 6.69E-06          | 0.04          | 38.35        |
| 0.33      | 0.0246            | 1.01                   | 5.58E-06          | -             | -            |
| 0.36      | 0.0212            | 2.2                    | 5.02E-06          | 0.01          | 41.88        |
| 0.42      | 0.0052            | 0.13                   | 3.91E-06          | -             | -            |
| 0.46      | 0.0023            | 0.45                   | 3.36E-06          | 0.01          | 34.07        |
| 0.52      | 0.0013            | 0.05                   | 1.69E-06          | -             | -            |
| 0.55      | 0.0032            | 0.11                   | 1.13E-06          | 0.01          | 29.52        |
| 0.62      | 0.001             | 0.09                   | 5.78E-07          | -             | -            |
| 0.65      | 0.0012            | 0.08                   | 2.22E-08          | 0             | 30.49        |

- **Training Loss**: The training loss decreases consistently, indicating the model is learning well.
- **Eval Loss**: The evaluation loss decreases significantly, showing that the model is generalizing well to unseen data.
- **Eval WER**: The Word Error Rate (WER) decreases over the epochs, indicating the model is getting better at transcribing code-switched Bengali-English speech.

## The Fine-Tune Process is Below 🛠️

### Clone the Repository

Open a terminal on your machine and use the following command to clone the repository:

```bash
git clone https://github.com/pr0mila/MediBeng-Whisper-Tiny.git
cd MediBeng-Whisper-Tiny
```


### Setup Instructions

#### Create a Conda Environment and Install Required Packages

To set up the environment, you can use one of the following commands:

```bash
conda create --name med-whisper-tiny python=3.9
conda activate med-whisper-tiny
```
Run the following command to install the packages listed in the requirements.txt file:
```bash
pip install -r requirements.txt
```
Or, install the packages
```bash
pip install torch transformers datasets librosa evaluate soundfile tensorboard jiwer accelerate
```
### Configuration Setup

The configuration parameters for the model, dataset, and repository are defined in the `config/config.py` file. For translated transcription, make sure to update the `LANGUAGE` and `TASK` variables as follows:

```python
MODEL_NAME = "openai/whisper-tiny"
LANGUAGE = "English"
TASK = "translate"
```
### Data Loading

The dataset is loaded and stored in the `data` folder, which is created by running the data processing code in the `data_loader.py` file. For training and testing, **20%** of the data from the dataset is used for both training and testing. This configuration is defined and controlled in the `data_loader.py` file.

### Training and Upload Model

To start training the model, run the following command:

```bash
python main.py
```
### Limitations
- **Accents**: The model may struggle with very strong regional accents or non-native speakers of Bengali and English.
- **Specialized Terms**: The model may not perform well with highly specialized medical terms or out-of-domain speech.
- **Multilingual Support**: While the model is designed for Bengali and English, other languages are not supported.

### Ethical Considerations
- **Biases**: The training data may contain biases based on the demographics of the speakers, such as gender, age, and accent.
- **Misuse**: Like any ASR system, this model could be misused to create fake transcripts of audio recordings, potentially leading to privacy and security concerns.
- **Fairness**: Ensure the model is used in contexts where fairness and ethical considerations are taken into account, particularly in clinical environments.

### License

This model is based on the **Whisper-Tiny** model by [OpenAI](https://huggingface.co/openai/whisper-tiny) available on [Hugging Face](https://huggingface.co). The original model is licensed under the **Apache-2.0** license.

This fine-tuned version, **Medibeng Whisper-Tiny**, was trained on a **code-switched Bengali-English** dataset for use in clinical settings and is also shared under the **Apache-2.0** license.
See the [LICENSE](LICENSE) file for more details.

#### Terms and Conditions
- You are free to use, modify, and distribute the model, as long as you comply with the conditions of the Apache License 2.0.
- You must provide attribution, including a reference to this model card and the repository when using or distributing the model.
- You cannot use the model for unlawful purposes or in any manner that infringes on the rights of others.

For more details, please review the full [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).

## Citation for Research Use

If you use **Medibeng Whisper-Tiny** or the **MediBeng** dataset for your research or project, please cite the following:

#### For **Medibeng Whisper-Tiny** Model (Fine-Tuned Model):

```bibtex
@misc{pr0mila2025medibengwhisper,
  author = {Promila Ghosh},
  title = {Medibeng Whisper-Tiny: Code-Switched Bengali-English Speech Translation for Clinical Settings},
  year = {2025},
  howpublished = {\url{https://huggingface.co/pr0mila-gh0sh/MediBeng-Whisper-Tiny}},
}
```
#### For MediBeng Dataset:
```bibtex
@dataset{pr0mila2025mediBengdataset,
  author = {Promila Ghosh},
  title = {MediBeng: Code-Switched Bengali-English Dataset for Clinical Transcription and Translation},
  year = {2025},
  howpublished = {\url{https://huggingface.co/datasets/pr0mila-gh0sh/MediBeng}},
}



