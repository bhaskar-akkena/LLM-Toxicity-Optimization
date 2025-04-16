from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import os


# Function to initialize the LLAMA model and tokenizer
def initialize_model(model_name="huggyllama/llama-7b"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model


# Function to generate text based on the prompt
def generate_text(prompt, model_name="huggyllama/llama-7b"):
    tokenizer, model = initialize_model(model_name)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs['input_ids']

    # Generate text from the input
    with torch.no_grad():
        outputs = model.generate(input_ids, max_length=150, num_return_sequences=1)

    # Decode the output text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text


# Function to save the generated response to a JSON file
def save_response(prompt, response, output_file_path):
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    response_data = {
        "prompt": prompt,
        "generated_response": response
    }

    with open(output_file_path, 'a') as file:
        json.dump(response_data, file)
        file.write("\n")  # New line for each response entry


# Main function to call the generation process
def main():
    prompt = "What is the capital of France?"  # Example prompt
    response = generate_text(prompt)  # Generate response from LLAMA

    # Define the path where responses will be saved
    output_file_path = "../../outputs/responses/generated_responses.json"

    # Save the generated response
    save_response(prompt, response, output_file_path)
    print(f"Response saved: {response}")


if __name__ == "__main__":
    main()
