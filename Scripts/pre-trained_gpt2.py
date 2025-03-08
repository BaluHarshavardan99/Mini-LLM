import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from datasets import load_dataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load and preprocess Wikipedia dataset
def load_and_preprocess_data():
    dataset = load_dataset("wikipedia", "20220301.en", split="train[:50%]", trust_remote_code=True)  
    print(f"Loaded dataset with {len(dataset)} samples.")

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token  

    # Tokenize the dataset
    def tokenize_function(examples):
        tokenized_output = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)
        tokenized_output["labels"] = tokenized_output["input_ids"].copy()  
        return tokenized_output

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    return tokenized_dataset, tokenizer

# Load a pre-trained GPT-2 model
def load_pretrained_gpt2():
    # Load the pre-trained GPT-2 model
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.to(device) 

# Fine-tune the model
def train_model(tokenized_dataset, tokenizer):
    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./fine-tuned-gpt2-wikipedia", 
        overwrite_output_dir=True,
        num_train_epochs=5,  
        per_device_train_batch_size=8,  
        save_steps=500,
        save_total_limit=2,
        logging_dir="./logs",
        logging_steps=100,
        learning_rate=5e-5, 
        warmup_steps=500,
        weight_decay=0.01,
        fp16=True, 
        gradient_accumulation_steps=4, 
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

    trainer.train()

    # Save the fine-tuned model and tokenizer
    trainer.save_model("./fine-tuned-gpt2-wikipedia")
    tokenizer.save_pretrained("./fine-tuned-gpt2-wikipedia")

# Save model architecture details
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

# Generate text for multiple prompts and save their outputs
# def generate_and_save_texts(prompts, model, tokenizer, num_texts_per_prompt=3, output_file="generated_texts.txt"):
#     model.to(device)  # Ensure model is on GPU
#     model.eval()  # Set model to evaluation mode

#     with open(output_file, "w") as f:
#         for prompt_idx, prompt in enumerate(prompts):
#             f.write(f"Prompt {prompt_idx + 1}: {prompt}\n")
#             print(f"Processing Prompt {prompt_idx + 1}: {prompt}")

#             for text_idx in range(num_texts_per_prompt):
#                 inputs = tokenizer(prompt, return_tensors="pt").to(device)
#                 outputs = model.generate(
#                     inputs.input_ids,
#                     max_length=100,
#                     num_beams=5,  
#                     no_repeat_ngram_size=2,
#                     early_stopping=True,
#                 )
#                 generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
#                 f.write(f"Generated Text {text_idx + 1}:\n{generated_text}\n\n")
#                 print(f"Generated Text {text_idx + 1}:\n{generated_text}\n")

#             f.write("\n" + "=" * 80 + "\n\n")  

if __name__ == "__main__":
    # Load and preprocess the dataset
    tokenized_dataset, tokenizer = load_and_preprocess_data()

    # Fine-tune the model
    train_model(tokenized_dataset, tokenizer)

    # Load the fine-tuned model
    model = GPT2LMHeadModel.from_pretrained("./fine-tuned-gpt2-wikipedia").to(device)  
    tokenizer = GPT2Tokenizer.from_pretrained("./fine-tuned-gpt2-wikipedia")

    # Save the model architecture details
    save_model_architecture(model, output_file="model_architecture.txt")

