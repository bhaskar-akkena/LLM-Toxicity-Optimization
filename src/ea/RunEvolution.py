## @file RunEvolution.py
#  @brief Main script for evolving LLM input prompts using mutation operators.
#
#  Loads a prompt population from JSON, applies variation operators, and saves
#  new prompt variants (offspring) back into the same population.
#  Terminates evolution per prompt_id if max toxicity of 1.0 is reached.

import json
import os
from EvolutionEngine import EvolutionEngine
import nltk

nltk.data.path.append('../../../.venv/nltk_data')


## @brief Main driver function to perform one generation of input prompt evolution.
#
#  Loads a population from JSON, performs tournament selection and variation
#  on each prompt_id group, appends offspring, and saves back the updated population.
## main function
# @brief Main driver function to perform one generation of input prompt evolution.  Loads a population from JSON, performs tournament selection and variation on each prompt_id group, appends offspring, and saves back the updated population.
# @return None
def main():
    population_path = "../../outputs/Population.json"
    output_path = "../../outputs/Population_EA.json"

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
                tox_values = [g.get("toxicity_score") for g in prompt_genomes if g.get("toxicity_score") is not None]

                if tox_values:
                    avg_tox = sum(tox_values) / len(tox_values)
                    if len(tox_values) >= 10 and avg_tox >= 1.0:
                        print(f"Prompt group {current_prompt_id} reached average toxicity of 1.0. Skipping.")
                        prompt_genomes = []
                        current_prompt_id = this_prompt_id
                        continue

            parents = engine.tournament_selection(prompt_genomes, k=5)
            offspring = engine.generate_offspring(parents, num_offspring_per_operator=3)
            new_offspring.extend(offspring)

            prompt_genomes = []
            current_prompt_id = this_prompt_id

        prompt_genomes.append(genome)

        if prompt_genomes:
            tox_values = [g.get("toxicity_score") for g in prompt_genomes if g.get("toxicity_score") is not None]
            if tox_values and len(tox_values) >= 10:
                avg_tox = sum(tox_values) / len(tox_values)
                if len(tox_values) >= 10 and avg_tox >= 1.0:
                    print(f"Prompt group {current_prompt_id} reached average toxicity of 1.0. Skipping.")
                else:
                    parents = engine.tournament_selection(prompt_genomes, k=5)
                    offspring = engine.generate_offspring(parents, num_offspring_per_operator=3)
                    new_offspring.extend(offspring)
            else:
                parents = engine.tournament_selection(prompt_genomes, k=5)
                offspring = engine.generate_offspring(parents, num_offspring_per_operator=3)
                new_offspring.extend(offspring)

    print(f"Generated {len(new_offspring)} new offspring.")

    population.extend(new_offspring)
    population.sort(key=lambda g: (
        g["prompt_id"],
        -(g["toxicity_score"] if isinstance(g.get("toxicity_score"), (int, float)) else 0.0),
        -g["generation"]
    ))

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(population, f, indent=4)

    print(f"Updated population with offspring saved to {output_path}")


if __name__ == "__main__":
    main()