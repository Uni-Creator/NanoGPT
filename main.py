import torch
from trainer import BigramLanguageModel, stoi, decode  # Ensure model.py has these

# Load model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = BigramLanguageModel()
model.load_state_dict(torch.load("model.pth", map_location=device))
model.to(device)
model.eval()

def generate_text(prompt, max_new_tokens=200):
    """Generate text based on user input."""
    prompt_encoded = torch.tensor([[stoi[c] for c in prompt]], dtype=torch.long, device=device)
    generated_indices = model.generate(prompt_encoded, max_new_tokens)
    return decode(generated_indices[0].tolist())

# Take user input
user_input = input("Enter a prompt: ")
generated_text = generate_text(user_input)
print("\nGenerated Text:\n", generated_text)