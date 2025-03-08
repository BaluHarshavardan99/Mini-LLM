import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from datasets import load_dataset

# Check if GPU is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Step 1: Load and preprocess Wikipedia dataset
def load_and_preprocess_data():
    # Load a subset of Wikipedia (1-2 GB)
    dataset = load_dataset("wikipedia", "20220301.en", split="train[:50%]", trust_remote_code=True)  # Adjust subset size as needed
    print(f"Loaded dataset with {len(dataset)} samples.")

    # Load a pre-trained BPE tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token  # Set padding token

    # Tokenize the dataset
    def tokenize_function(examples):
        tokenized_output = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)
        tokenized_output["labels"] = tokenized_output["input_ids"].copy()  # Labels for causal LM
        return tokenized_output

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    return tokenized_dataset, tokenizer

# Step 2: Load a pre-trained GPT-2 model
def load_pretrained_gpt2():
    # Load the pre-trained GPT-2 model
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.to(device)  # Move model to GPU
    return model

# Step 3: Fine-tune the model
def train_model(tokenized_dataset, tokenizer):
    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./fine-tuned-gpt2-wikipedia",  # Directory to save the fine-tuned model
        overwrite_output_dir=True,
        num_train_epochs=5,  # Fine-tune for 1 epoch (adjust as needed)
        per_device_train_batch_size=8,  # Adjust based on GPU memory
        save_steps=500,
        save_total_limit=2,
        logging_dir="./logs",
        logging_steps=100,
        learning_rate=5e-5,  # Smaller learning rate for fine-tuning
        warmup_steps=500,
        weight_decay=0.01,
        fp16=True,  # Use mixed precision
        gradient_accumulation_steps=4,  # Accumulate gradients to simulate larger batch size
    )

    # Initialize the pre-trained model
    model = load_pretrained_gpt2()

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
    )

    # Fine-tune the model
    trainer.train()

    # Save the fine-tuned model and tokenizer
    trainer.save_model("./fine-tuned-gpt2-wikipedia")
    tokenizer.save_pretrained("./fine-tuned-gpt2-wikipedia")

# Step 4: Save model architecture details
def save_model_architecture(model, output_file="model_architecture.txt"):
    with open(output_file, "w") as f:
        f.write(f"Model Type: {model.config.model_type}\n")
        f.write(f"Number of Layers: {model.config.n_layer}\n")
        f.write(f"Hidden Size: {model.config.n_embd}\n")
        f.write(f"Number of Attention Heads: {model.config.n_head}\n")
        f.write(f"Vocabulary Size: {model.config.vocab_size}\n")
        f.write(f"Max Position Embeddings: {model.config.n_positions}\n")
        f.write(f"Total Parameters: {sum(p.numel() for p in model.parameters())}\n")

    print(f"Model architecture details saved to {output_file}")

# Step 5: Generate text for multiple prompts and save their outputs
def generate_and_save_texts(prompts, model, tokenizer, num_texts_per_prompt=3, output_file="generated_texts.txt"):
    model.to(device)  # Ensure model is on GPU
    model.eval()  # Set model to evaluation mode

    with open(output_file, "w") as f:
        for prompt_idx, prompt in enumerate(prompts):
            f.write(f"Prompt {prompt_idx + 1}: {prompt}\n")
            print(f"Processing Prompt {prompt_idx + 1}: {prompt}")

            for text_idx in range(num_texts_per_prompt):
                inputs = tokenizer(prompt, return_tensors="pt").to(device)
                outputs = model.generate(
                    inputs.input_ids,
                    max_length=100,
                    num_beams=5,  # Use beam search for better coherence
                    no_repeat_ngram_size=2,
                    early_stopping=True,
                )
                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                f.write(f"Generated Text {text_idx + 1}:\n{generated_text}\n\n")
                print(f"Generated Text {text_idx + 1}:\n{generated_text}\n")

            f.write("\n" + "=" * 80 + "\n\n")  # Separator between prompts

if __name__ == "__main__":
    # Load and preprocess the dataset
    tokenized_dataset, tokenizer = load_and_preprocess_data()

    # Fine-tune the model
    train_model(tokenized_dataset, tokenizer)

    # Load the fine-tuned model
    model = GPT2LMHeadModel.from_pretrained("./fine-tuned-gpt2-wikipedia").to(device)  # Move model to GPU
    tokenizer = GPT2Tokenizer.from_pretrained("./fine-tuned-gpt2-wikipedia")

    # Save the model architecture details
    save_model_architecture(model, output_file="model_architecture.txt")

    # Define multiple prompts for testing
    prompts = [
        "The history of the Roman Empire",
        "The development of the internet",
        "The causes and effects of World War II",
        "The theory of relativity explained",
        "The evolution of human language",
        "The structure and function of DNA",
        "The impact of the Industrial Revolution on society",
        "The discovery of penicillin and its significance",
        "The rise and fall of the Byzantine Empire",
        "The principles of quantum mechanics",
        "The history of the printing press",
        "The role of photosynthesis in the ecosystem",
        "The cultural significance of the Renaissance",
        "The history of space exploration",
        "The effects of globalization on world economies",
        "The history of the Olympic Games",
        "The science behind climate change",
        "The history of the United Nations",
        "The invention of the telephone and its impact",
        "The history of the Silk Road",
    ]

    # Generate and save texts for multiple prompts
    generate_and_save_texts(prompts, model, tokenizer, num_texts_per_prompt=3, output_file="generated_texts.txt")
