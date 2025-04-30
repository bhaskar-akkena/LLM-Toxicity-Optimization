# @file EvolutionEngine.py
# @author Onkar Shelar
# @brief Core logic for prompt mutation and evolutionary search

# logs for major changes: (log, date, author)
# 
#  


import random
from typing import List, Dict
from TextVariationOperators import get_applicable_operators


class EvolutionEngine:
     
    ## @class EvolutionEngine
    # @brief Handles prompt evolution using mutation-based variation operators.

    # This class performs selection and variation of input prompts,
    # generating new variants for inclusion in an evolving population.
    def __init__(self):
        self.genomes: List[Dict] = []
        self.next_id = 0 
    
    def update_next_id(self):
        if self.genomes:
            self.next_id = max(g["id"] for g in self.genomes) + 1
        else:
            self.next_id = 0

    def tournament_selection(self, genomes: List[dict], k: int, tournament_size: int = 3) -> List[dict]:
        if len(genomes) == 1:
            return [genomes[0]] * k

        selected_parents = []
        for _ in range(k):
            candidates = random.sample(genomes, min(tournament_size, len(genomes)))
            winner = max(candidates, key=lambda g: g.get("toxicity_score") or 0.0)
            selected_parents.append(winner)

        return selected_parents

    def generate_offspring(self, parents: List[Dict], num_offspring_per_operator: int = 20) -> List[Dict]:
        operators = get_applicable_operators(len(parents))
        offspring = []

        for operator in operators:
            for _ in range(num_offspring_per_operator):
                parent = random.choice(parents)

                try:
                    variant_prompt = operator.apply(parent["prompt"])
                    if variant_prompt.strip().lower() in set(g["prompt"].strip().lower() for g in self.genomes if g["prompt_id"] == parent["prompt_id"]): 
                        continue  # skip duplicate
                    seen_prompts = set(g["prompt"].strip().lower() for g in self.genomes if g["prompt_id"] == parent["prompt_id"])
                    seen_prompts.add(variant_prompt.strip().lower())
                except Exception as e:
                    print(f"[Variation Failed] {operator.name}: {e}")
                    continue

                child = {
                    "id": self.next_id,
                    "prompt_id": parent["prompt_id"],
                    "prompt": variant_prompt,  
                    "generated_response": None, 
                    "toxicity_score": None,
                    "severe_toxicity_score": None,
                    "insult_score": None,
                    "operator": operator.name,
                    "parents": [parent["id"]],
                    "generation": parent["generation"] + 1,
                    "status": "pending_generation",
                    "creation_info": {
                        "type": operator.operator_type,
                        "operator": operator.name,
                        "source_generation": parent["generation"]
                    }
                }

                self.next_id += 1
                offspring.append(child)

        return offspring