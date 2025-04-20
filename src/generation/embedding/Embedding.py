from transformers import AutoModelForCausalLM
import torch
import json
import numpy as np
import time

# Function to initialize the LLAMA model
def initialize_model(model_name="huggyllama/llama-7b"):
    print("Initializing model...")
    start_time = time.time()
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name)
        print(f"Model loaded in {time.time() - start_time:.2f} seconds.")
    except Exception as e:
        print(f"Error loading model {model_name}: {str(e)}")
        raise
    return model

# Step 1: Generating a random embedding using torch
embedding_size = 4096
random_embedding = torch.tensor(np.random.rand(1, 1, embedding_size), dtype=torch.float32)  # Shape (batch_size, seq_length, embedding_size)

# Step 2: Pass random embedding directly to the model using `inputs_embeds`
def generate_output_from_embedding(model, random_embedding):
    """
    This function directly passes an embedding to the model (without tokenization).
    """
    with torch.no_grad():
        # Pass the embedding directly to the model using `inputs_embeds`
        outputs = model(inputs_embeds=random_embedding)

    # Get the logits (raw predictions from the model)
    logits = outputs.logits

    # Extract the most likely token ID from the logits
    predicted_token_ids = torch.argmax(logits, dim=-1)

    # Decode the token IDs into text (Note: In practice, you would use a tokenizer here,
    # but since you don't want to use a tokenizer, we'll skip this step)
    output_text = f"Predicted token ID: {predicted_token_ids.item()}"

    return output_text

# Initialize the model
model_name = "huggyllama/llama-7b"
model = initialize_model(model_name)

# Generate output directly from the random embedding
output_text = generate_output_from_embedding(model, random_embedding)
print(f"Generated Output: {output_text}")

# Save the output to a JSON file
response_data = {
    "response": output_text,
    "random_embedding": random_embedding.squeeze().tolist(),  # Convert tensor to list for JSON serialization
}

json_file_path = "generated_responses.json"
with open(json_file_path, "w") as f:
    json.dump(response_data, f, indent=4)

print(f"Response data saved to '{json_file_path}'.")
