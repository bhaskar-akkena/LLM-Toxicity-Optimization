import json
import os

def load_and_initialize_population(input_path: str, output_path: str):
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        raise RuntimeError(f"Error loading file: {e}")

    prompt_to_id = {}
    prompt_counter = 0
    genomes = []

    for idx, entry in enumerate(data):
        prompt = entry["prompt"]

        if prompt not in prompt_to_id:
            prompt_to_id[prompt] = prompt_counter
            prompt_counter += 1

        genome = {
            "id": idx,
            "prompt_id": prompt_to_id[prompt]+1,
            "prompt": prompt,
            "generated_response": entry["generated_response"],
            "toxicity_score": entry.get("toxicity_score"),
            "severe_toxicity_score": entry.get("severe_toxicity_score"),
            "insult_score": entry.get("insult_score"),
            "operator": "original",
            "parents": [],
            "generation": 0,
            "status": "scored"
        }

        genomes.append(genome)

    sorted_genomes = sorted(genomes, key=lambda g: g["prompt_id"])

    os.makedirs(os.path.dirname(output_path), exist_ok=True) if os.path.dirname(output_path) else None

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(sorted_genomes, f, indent=4)

    print(f"Initialized {len(sorted_genomes)} genomes from {len(prompt_to_id)} unique prompts.")
    print(f"Saved sorted population to {output_path}")


if __name__ == "__main__":
    input_file = "../../../outputs/responses/generated_responses_with_scores1.json"
    output_file = "../../../outputs/Population.json"
    load_and_initialize_population(input_file, output_file)