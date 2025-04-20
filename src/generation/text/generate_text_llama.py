from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import os
import time

# Function to initialize the LLAMA model and tokenizer
def initialize_model(model_name="huggyllama/llama-7b"):
    print("Initializing model...")
    start_time = time.time()
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, legacy=False)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        print(f"Model and tokenizer loaded in {time.time() - start_time:.2f} seconds.")
    except Exception as e:
        print(f"Error loading model {model_name}: {str(e)}")
        raise
    return tokenizer, model

# Function to clean the response by removing the question part if repeated
def clean_response(response, prompt):
    """Remove the question from the generated response if it's repeated."""
    if response.startswith(prompt):
        return response[len(prompt):].strip()  # Remove the question part if it repeats
    return response

# Function to generate text based on the prompt
def generate_text(prompt, model_name="huggyllama/llama-7b"):
    print(f"Generating text for prompt: {prompt}")
    start_time = time.time()
    try:
        tokenizer, model = initialize_model(model_name)
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs['input_ids']

        # Generate text from the input
        print("Generating response from model...")
        with torch.no_grad():
            outputs = model.generate(input_ids, max_length=150, num_return_sequences=1)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Clean the generated response by removing the question
        cleaned_text = clean_response(generated_text, prompt)
        cleaned_text= cleaned_text.replace("\n", " ").strip()
        print(f"Response generated in {time.time() - start_time:.2f} seconds.")
    except Exception as e:
        print(f"Error during text generation: {str(e)}")
        raise
    return cleaned_text

# Function to save the generated response to a JSON file
def save_response(prompt, response, output_file_path):
    print(f"Saving response to {output_file_path}...")
    start_time = time.time()
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

        # Open the file in append mode (text mode)
        with open(output_file_path, 'a') as file:
            # Check if the file is empty, if so, start the JSON array
            if file.tell() == 0:
                file.write("[\n")  # Start of the JSON array

            # Append the generated response as a JSON object
            response_data = {
                "prompt": prompt,
                "generated_response": response
            }
            json.dump(response_data, file)
            file.write(",\n")  # Add a comma and new line after each entry

        # Close the JSON array after the last entry
        with open(output_file_path, 'r+') as file:
            file.seek(-2, os.SEEK_END)  # Move to the last comma
            file.truncate()  # Remove the last comma
            file.write("\n]")  # Close the JSON array

        print(f"Response saved in {time.time() - start_time:.2f} seconds.")
    except Exception as e:
        print(f"Error saving response: {str(e)}")
        raise

# Main function to call the generation process
def main():
    prompt = "What is the capital of France?"  # Example prompt
    print(f"Starting text generation for prompt: {prompt}")
    start_time = time.time()

    try:
        response = generate_text(prompt)  # Generate response from LLAMA
        output_file_path = "../../../outputs/responses/generated_responses.json"
        save_response(prompt, response, output_file_path)
        print(f"Total execution time: {time.time() - start_time:.2f} seconds.")
    except Exception as e:
        print(f"Error in main process: {str(e)}")

if __name__ == "__main__":
    main()
