import json
from collections import defaultdict

def clean_mutated_prompts(input_file: str, json_out: str, txt_out: str):
    with open(input_file, "r", encoding="utf-8") as f:
        population = json.load(f)

    # Step 1: Filter entries with status == "pending_generation"
    pending_mutations = [entry for entry in population if entry.get("status") == "pending_generation"]

    # Step 2: Group by prompt_id and remove duplicate prompts
    grouped_by_prompt_id = defaultdict(list)
    for entry in pending_mutations:
        grouped_by_prompt_id[entry["prompt_id"]].append(entry)

    unique_mutated_prompts = []
    seen_prompts = set()
    for prompt_id, group in grouped_by_prompt_id.items():
        for entry in group:
            prompt_text = entry["prompt"].strip()
            if prompt_text not in seen_prompts:
                seen_prompts.add(prompt_text)
                unique_mutated_prompts.append(entry)

    # Step 3: Save to new JSON file
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(unique_mutated_prompts, f, indent=2)

    # Step 4: Save all unique questions to txt file as a Python list
    questions_list = sorted(seen_prompts)
    with open(txt_out, "w", encoding="utf-8") as f:
        f.write("questions = [\n")
        for q in questions_list:
            f.write(f'    "{q}",\n')
        f.write("]\n")

    print(f"Saved unique mutated prompts to: {json_out}")
    print(f"Saved unique prompt text list to: {txt_out}")


if __name__ == "__main__":
    input_json = "Population_EA.json"
    output_json = "Population_EA_Mutated_Unique.json"
    output_txt = "mutated_prompt_list.txt"
    clean_mutated_prompts(input_json, output_json, output_txt)
