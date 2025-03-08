# Mini-LLM

# GPT-2 Models for Text Generation

This repository contains three GPT-2 models trained on Wikipedia data:
1. **Minimal GPT-2 (10% Data)**: A smaller GPT-2 model trained from scratch on 10% of the Wikipedia dataset.
2. **Minimal GPT-2 (30% Data)**: A smaller GPT-2 model trained from scratch on 30% of the Wikipedia dataset.
3. **GPT-2 Pre-trained**: A pre-trained GPT-2 model fine-tuned on 50% of the Wikipedia dataset.

The repository includes:
- Python scripts for training and evaluating the models.
- Instructions for downloading the model checkpoints from Google Drive.
- Instructions for setting up and running the models.

---

## **Requirements**

To run the scripts and use the checkpoints, you need the following dependencies:

- Python 3.8 or higher
- PyTorch
- Hugging Face Transformers
- Datasets library
- NLTK
- NumPy

Install the dependencies using the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
