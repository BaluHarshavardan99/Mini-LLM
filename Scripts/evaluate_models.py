from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch


model_paths = {
    "10%_from_scratch": "/checkpoint-100915",
    "30%_from_scratch": "/checkpoint-302750",
    "pretrained_model": "/checkpoint-201830"
}


def calculate_perplexity(model, tokenizer, text):

    input_ids = tokenizer.encode(text, return_tensors="pt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_ids = input_ids.to(device)
    model.to(device)

    # Calculate perplexity
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
        perplexity = torch.exp(loss).item()

    return perplexity

# Evaluate each model
input_text = "The cat was found under the bed"  # Replace with your input text

for model_name, model_path in model_paths.items():
    print(f"Loading {model_name} from {model_path}...")
    
    # Load model and tokenizer
    model = GPT2LMHeadModel.from_pretrained(model_path)
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    
    # Calculate perplexity
    perplexity = calculate_perplexity(model, tokenizer, input_text)
    print(f"Perplexity for {model_name}: {perplexity}\n")
