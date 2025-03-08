Here’s a concise and clear **README.md** content focused on accessing the checkpoints and running the models as code. This version is streamlined for users who want to quickly get started with your models.

---

### **README.md**

```markdown
# GPT-2 Models for Text Generation

This repository contains three GPT-2 models trained on Wikipedia data:
1. **Minimal GPT-2 (10% Data)**: A smaller GPT-2 model trained from scratch on 10% of the Wikipedia dataset.
2. **Minimal GPT-2 (30% Data)**: A smaller GPT-2 model trained from scratch on 30% of the Wikipedia dataset.
3. **GPT-2 Pre-trained**: A pre-trained GPT-2 model fine-tuned on 50% of the Wikipedia dataset.

The repository includes:
- Python scripts for training and evaluating the models.
- Instructions for downloading and using the model checkpoints from Google Drive.

---

## **Quick Start**

### **1. Install Dependencies**

Install the required Python packages:
```bash
pip install torch transformers datasets nltk numpy evaluate
```

Download NLTK data (if not already downloaded):
```bash
python -m nltk.downloader punkt
```

---

### **2. Download Model Checkpoints**

The model checkpoints are hosted on Google Drive due to their size. Download them using the links below:

- [Minimal GPT-2 (10% Data)](https://drive.google.com/your-link-here)
- [Minimal GPT-2 (30% Data)](https://drive.google.com/your-link-here)
- [Pre-trained GPT-2](https://drive.google.com/your-link-here)

After downloading, extract the checkpoints into a folder named `checkpoints/` in the root of the repository. The folder structure should look like this:
```
checkpoints/
├── minimal_gpt2_10/          # Checkpoint for Minimal GPT-2 (10% Data)
├── minimal_gpt2_30/          # Checkpoint for Minimal GPT-2 (30% Data)
└── pretrained_gpt2/          # Checkpoint for Pre-trained GPT-2
```

---

### **3. Load a Checkpoint in Code**

To load a checkpoint and generate text, use the following code:

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load the model and tokenizer
model = GPT2LMHeadModel.from_pretrained("checkpoints/minimal_gpt2_10")
tokenizer = GPT2Tokenizer.from_pretrained("checkpoints/minimal_gpt2_10")

# Generate text
prompt = "The history of the Roman Empire"
input_ids = tokenizer.encode(prompt, return_tensors="pt")
output = model.generate(input_ids, max_length=50, num_beams=5, no_repeat_ngram_size=2, early_stopping=True)

# Decode and print the generated text
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)
```

---

### **4. Evaluate the Models**

To evaluate all three models and generate text for a set of prompts, run the `evaluate_models.py` script:

```bash
cd scripts
python evaluate_models.py
```

This script will:
- Load the checkpoints for all three models.
- Calculate perplexity for each model.
- Generate text for a set of predefined prompts.

---

### **5. Train the Models (Optional)**

If you want to retrain the models, use the following scripts:

#### **Minimal GPT-2 (10% Data)**
```bash
python train_minimal_gpt2.py --data_percent 10
```

#### **Minimal GPT-2 (30% Data)**
```bash
python train_minimal_gpt2.py --data_percent 30
```

#### **Pre-trained GPT-2**
```bash
python train_pretrained_gpt2.py
```

---

## **Repository Structure**

```
gpt2-wikipedia/
├── checkpoints/                # Folder for model checkpoints (download from Google Drive)
├── scripts/
│   ├── train_minimal_gpt2.py   # Script to train Minimal GPT-2 from scratch
│   ├── train_pretrained_gpt2.py # Script to fine-tune Pre-trained GPT-2
│   └── evaluate_models.py      # Script to evaluate the models
├── requirements.txt            # List of dependencies
└── README.md                   # This file
```

---

## **License**

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
```

---

### **Key Features of the README**
1. **Clear Instructions**: Step-by-step guidance for downloading checkpoints, installing dependencies, and running the models.
2. **Code Examples**: Ready-to-use code snippets for loading checkpoints and generating text.
3. **Script Usage**: Instructions for evaluating and training the models using the provided scripts.
4. **Google Drive Integration**: Links to download large checkpoints hosted on Google Drive.

---

### **requirements.txt**

```plaintext
torch>=1.10.0
transformers>=4.18.0
datasets>=2.0.0
nltk>=3.7
numpy>=1.21.0
evaluate>=0.3.0
```

---

This README is designed to be user-friendly and provides all the necessary information to quickly get started with your models. Let me know if you need further adjustments!
