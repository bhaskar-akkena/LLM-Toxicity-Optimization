## @file RunEvolution.py
#  @brief Main script for evolving LLM input prompts using mutation operators.
#
#  Loads a prompt population from JSON, applies variation operators, and saves
#  new prompt variants (offspring) back into the same population.
#  Terminates evolution per prompt_id if max toxicity of 1.0 is reached.

import json
import os
from EvolutionEngine import EvolutionEngine

## @brief Main driver function to perform one generation of input prompt evolution.
#
#  Loads a population from JSON, performs tournament selection and variation
#  on each prompt_id group, appends offspring, and saves back the updated population.
def main():
    population_path = "../../outputs/Population.json"

    if not os.path.exists(population_path):
        raise FileNotFoundError(f"{population_path} not found.")

    with open(population_path, 'r', encoding='utf-8') as f:
        population = json.load(f)

    print(f"Loaded {len(population)} genomes from {population_path}.")

    engine = EvolutionEngine()
    engine.genomes = population
    engine.update_next_id()

    current_prompt_id = None
    prompt_genomes = []
    new_offspring = []

    engine.genomes.sort(key=lambda g: g["prompt_id"])

    for genome in engine.genomes:
        this_prompt_id = genome["prompt_id"]

        if current_prompt_id is None:
            current_prompt_id = this_prompt_id

        if this_prompt_id != current_prompt_id:
            if prompt_genomes:
                max_toxicity = max((g.get("toxicity_score") or 0.0) for g in prompt_genomes)
                
                if max_toxicity >= 1.0:
                    print(f"Skipping evolution for prompt_id {current_prompt_id} (max toxicity = 1.0)")
                else:
                    parents = engine.tournament_selection(prompt_genomes, k=5)
                    offspring = engine.generate_offspring(parents, num_offspring_per_operator=2)
                    new_offspring.extend(offspring)

            prompt_genomes = []
            current_prompt_id = this_prompt_id

        prompt_genomes.append(genome)

    if prompt_genomes:
        parents = engine.tournament_selection(prompt_genomes, k=5)
        offspring = engine.generate_offspring(parents, num_offspring_per_operator=2)
        new_offspring.extend(offspring)

    print(f"Generated {len(new_offspring)} new offspring.")

    population.extend(new_offspring)
    population.sort(key=lambda g: g["prompt_id"])

    with open(population_path, 'w', encoding='utf-8') as f:
        json.dump(population, f, indent=4)

    print(f"Updated population with offspring saved to {population_path}")


if __name__ == "__main__":
    main()