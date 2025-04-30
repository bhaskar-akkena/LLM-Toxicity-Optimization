## @file initialize_population.py
#  @author Onkar Shelar
#  @brief Initializes a structured genome population JSON from scored prompt-response pairs.
#
#  This script processes a scored responses JSON file to assign unique prompt IDs,
#  and formats the data into a genome structure compatible with evolutionary mutation pipelines.
#  The output is sorted and saved for use in prompt evolution.

import json
import os


def load_and_initialize_population(input_path: str, output_path: str):
    """
    Loads data from a JSON file, processes it, and initializes a population of genomes with associated scores.

    This function loads the data from the provided `input_path`, processes each entry to assign a unique prompt ID,
    and creates a list of genomes with additional information such as toxicity scores, insult scores, etc.
    The resulting population is then saved to the specified `output_path`.

    Args:
        input_path (str): The path to the input JSON file containing the prompts and their associated responses and scores.
        output_path (str): The path to save the initialized population to a JSON file.

    Raises:
        RuntimeError: If there is an issue loading the input file.
    """
    try:
        # Load the input data from the JSON file
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        # Raise an error if the file could not be loaded
        raise RuntimeError(f"Error loading file: {e}")

    # Initialize a mapping for prompt IDs and a counter
    prompt_to_id = {}
    prompt_counter = 0
    genomes = []

    # Iterate over the loaded data and process each entry
    for idx, entry in enumerate(data):
        prompt = entry["prompt"]

        # Assign a unique ID to each unique prompt
        if prompt not in prompt_to_id:
            prompt_to_id[prompt] = prompt_counter
            prompt_counter += 1

        # Create a genome for each entry with relevant information
        genome = {
            "id": idx,
            "prompt_id": prompt_to_id[prompt] + 1,  # Increment by 1 to make prompt IDs 1-based
            "prompt": prompt,
            "generated_response": entry["generated_response"],
            "toxicity_score": entry.get("toxicity_score"),
            "severe_toxicity_score": entry.get("severe_toxicity_score"),
            "insult_score": entry.get("insult_score"),
            "operator": "original",  # The operator is set as "original" for now
            "parents": [],  # No parents initially
            "generation": 0,  # Initial generation is set to 0
            "status": "scored"  # Status is set to "scored" for all genomes
        }

        # Append the genome to the list of genomes
        genomes.append(genome)

    # Sort the genomes based on the prompt_id
    sorted_genomes = sorted(genomes, key=lambda g: g["prompt_id"])

    # Ensure the directory for the output file exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True) if os.path.dirname(output_path) else None

    # Save the sorted genomes to the output file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(sorted_genomes, f, indent=4)

    # Print confirmation that the population has been initialized and saved
    print(f"Initialized {len(sorted_genomes)} genomes from {len(prompt_to_id)} unique prompts.")
    print(f"Saved sorted population to {output_path}")


# Main execution if this script is run directly
if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.normpath(os.path.join(base_dir, '../../../outputs/responses/generated_responses_with_scores.json'))
    output_file = os.path.normpath(os.path.join(base_dir, '../../../outputs/Population.json'))
    load_and_initialize_population(input_file, output_file)
