import torch
from transformers import GPT2Tokenizer, GPT2Config, GPT2LMHeadModel, Trainer, TrainingArguments
from datasets import load_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Step 1: Load and preprocess Wikipedia dataset
def load_and_preprocess_data():
    dataset = load_dataset("wikipedia", "20220301.en", split="train[:30%]", trust_remote_code=True)  # Adjust subset size as needed
    print(f"Loaded dataset with {len(dataset)} samples.")

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token 

    # Tokenize the dataset
    def tokenize_function(examples):
        tokenized_output = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)
        tokenized_output["labels"] = tokenized_output["input_ids"].copy()  # Labels for causal LM
        return tokenized_output

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    return tokenized_dataset, tokenizer

# Define the minimal GPT-2 model
def create_minimal_gpt2():
    config = GPT2Config(
        vocab_size=50257,  
        n_positions=512, 
        n_embd=512,       
        n_layer=6,     
        n_head=8,        
        n_inner=2048,      
        activation_function="gelu",
        resid_pdrop=0.1,  
        embd_pdrop=0.1,
        attn_pdrop=0.1,
    )
    model = GPT2LMHeadModel(config)
    model.to(device)  
    return model

# Train the model
def train_model(tokenized_dataset, tokenizer):
    training_args = TrainingArguments(
        output_dir="./minimal-gpt2-wikipedia",
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

    # Initialize the model
    model = create_minimal_gpt2()

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
    )

    trainer.train()

    trainer.save_model("./minimal-gpt2-wikipedia")
    tokenizer.save_pretrained("./minimal-gpt2-wikipedia")

# # Generate text for multiple prompts and save their outputs
# def generate_and_save_texts(prompts, model, tokenizer, num_texts_per_prompt=3, output_file="generated_texts.txt"):
#     model.to(device)  
#     model.eval() 

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

# Save model architecture details
def save_model_architecture_details(model, output_file="model_architecture.txt"):
    with open(output_file, "w") as f:
        config = model.config
        f.write("Model Architecture Details\n")
        f.write("=" * 40 + "\n")
        f.write(f"Model Name: GPT-2 (Minimal Version)\n")
        f.write(f"Number of Parameters: {sum(p.numel() for p in model.parameters())}\n")
        f.write(f"Number of Layers: {config.n_layer}\n")
        f.write(f"Number of Attention Heads: {config.n_head}\n")
        f.write(f"Hidden Size: {config.n_embd}\n")
        f.write(f"Feedforward Dimension: {config.n_inner}\n")
        f.write(f"Maximum Sequence Length: {config.n_positions}\n")
        f.write(f"Vocabulary Size: {config.vocab_size}\n")
        f.write(f"Dropout Rate (Residual): {config.resid_pdrop}\n")
        f.write(f"Dropout Rate (Embedding): {config.embd_pdrop}\n")
        f.write(f"Dropout Rate (Attention): {config.attn_pdrop}\n")
        f.write("\n")

    print("Model architecture details saved successfully.")

if __name__ == "__main__":
    tokenized_dataset, tokenizer = load_and_preprocess_data()
    
    # Train the model
    train_model(tokenized_dataset, tokenizer)

    # Load the trained model
    model = GPT2LMHeadModel.from_pretrained("./minimal-gpt2-wikipedia").to(device) 
    tokenizer = GPT2Tokenizer.from_pretrained("./minimal-gpt2-wikipedia")

    # Save model architecture details
    save_model_architecture_details(model, output_file="model_architecture.txt")

   
