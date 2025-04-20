from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import os
import time

# Function to initialize the model and tokenizer
def initialize_model(model_name="deepseek-ai/deepseek-coder-1.3b-instruct"):
    print("Initializing model...")
    start_time = time.time()
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        print(f"Model and tokenizer loaded in {time.time() - start_time:.2f} seconds.")
    except Exception as e:
        print(f"Error loading model {model_name}: {str(e)}")
        raise
    return tokenizer, model

# Function to generate text based on the prompt
def generate_text(prompt, model_name="deepseek-ai/deepseek-coder-1.3b-instruct"):
    print(f"Generating text for prompt: {prompt}")
    start_time = time.time()
    try:
        tokenizer, model = initialize_model(model_name)
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)

        # Ensure attention mask is set
        attention_mask = inputs.get("attention_mask", None)
        if attention_mask is None:
            attention_mask = torch.ones_like(inputs['input_ids'])  # Use 1 for all non-padding tokens

        # Set pad_token_id if it's not defined
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

        # Move model and inputs to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        attention_mask = attention_mask.to(device)

        # Generate text from the input
        print("Generating response from model...")
        with torch.no_grad():
            outputs = model.generate(
                inputs['input_ids'],
                attention_mask=attention_mask,
                max_length=150,
                num_return_sequences=1,
                pad_token_id=pad_token_id  # Set pad_token_id explicitly
            )
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Response generated in {time.time() - start_time:.2f} seconds.")
    except Exception as e:
        print(f"Error during text generation: {str(e)}")
        raise
    return generated_text

# Function to save the generated response to a JSON file
def save_response(prompt, response, output_file_path):
    print(f"Saving response to {output_file_path}...")
    start_time = time.time()
    try:
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        response_data = {
            "prompt": prompt,
            "generated_response": response
        }
        with open(output_file_path, 'a') as file:
            json.dump(response_data, file)
            file.write("\n")  # New line for each response entry
        print(f"Response saved in {time.time() - start_time:.2f} seconds.")
    except Exception as e:
        print(f"Error saving response: {str(e)}")
        raise

# Main function to call the generation process
def main():
    prompt = "Why are some people stupid?"  # Example prompt
    print(f"Starting text generation for prompt: {prompt}")
    start_time = time.time()

    try:
        response = generate_text(prompt)  # Generate response from deepseek model
        output_file_path = "../../../outputs/responses/generated_responses.json"
        save_response(prompt, response, output_file_path)
        print(f"Total execution time: {time.time() - start_time:.2f} seconds.")
    except Exception as e:
        print(f"Error in main process: {str(e)}")

if __name__ == "__main__":
    main()
