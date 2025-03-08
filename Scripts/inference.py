import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Model paths (update these paths to point to your Google Drive or local paths)
model_paths = {
    "10%_from_scratch": "/minimal-gpt2-10/checkpoint-100915",
    "30%_from_scratch": "/minimal-gpt2-30/checkpoint-302750",
    "pretrained_model": "/fine-tuned-gpt2_pretrained/checkpoint-201830"
}

def load_model_and_tokenizer(checkpoint_path):
    """Load the fine-tuned GPT-2 model and tokenizer from the checkpoint."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPT2LMHeadModel.from_pretrained(checkpoint_path).to(device)
    tokenizer = GPT2Tokenizer.from_pretrained(checkpoint_path)
    return model, tokenizer, device

def generate_texts(prompts, model, tokenizer, device, num_texts_per_prompt=3, output_file="generated_texts.txt"):
    """Generate and save texts for given prompts."""
    model.eval()  # Set model to evaluation mode
    
    with open(output_file, "w") as f:
        for prompt_idx, prompt in enumerate(prompts):
            f.write(f"Prompt {prompt_idx + 1}: {prompt}\n")
            print(f"Processing Prompt {prompt_idx + 1}: {prompt}")

            for text_idx in range(num_texts_per_prompt):
                inputs = tokenizer(prompt, return_tensors="pt").to(device)
                outputs = model.generate(
                    inputs.input_ids,
                    max_length=300,
                    num_beams=5,  # Beam search for better coherence
                    no_repeat_ngram_size=2,
                    early_stopping=True,
                )
                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                f.write(f"Generated Text {text_idx + 1}:\n{generated_text}\n\n")
                print(f"Generated Text {text_idx + 1}:\n{generated_text}\n")

            f.write("\n" + "=" * 80 + "\n\n")  # Separator between prompts

if __name__ == "__main__":
    # List of prompts
    prompts = [
        "Describe the key events leading up to the fall of the Roman Empire, including the role of barbarian invasions, economic decline, and political instability.",
        "Explain the significance of the discovery of penicillin by Alexander Fleming and its impact on modern medicine.",
        "Discuss the development of the World Wide Web by Tim Berners-Lee and how it revolutionized information sharing and communication.",
    ]



    # Generate texts for each model
    for model_name, checkpoint_path in model_paths.items():
        print(f"Generating texts for {model_name}...")
        model, tokenizer, device = load_model_and_tokenizer(checkpoint_path)
        output_file = f"generated_texts_{model_name}.txt"  # Unique output file for each model
        generate_texts(prompts, model, tokenizer, device, output_file=output_file)
        print(f"Texts for {model_name} saved to {output_file}\n")

