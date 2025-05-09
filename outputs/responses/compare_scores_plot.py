import json
import matplotlib.pyplot as plt
import numpy as np

# Load files
with open("generated_responses_with_scores.json", "r") as f:
    original_data = json.load(f)

with open("generated_responses_mutated_with_scores.json", "r") as f:
    mutated_data = json.load(f)

# Helper to compute mean safely
def mean_score(data, key):
    return np.mean([entry[key] for entry in data if entry[key] is not None])

# Score types to compare
score_keys = ["toxicity_score", "severe_toxicity_score", "insult_score"]
original_means = [mean_score(original_data, key) for key in score_keys]
mutated_means = [mean_score(mutated_data, key) for key in score_keys]

# Bar plot setup
x = np.arange(len(score_keys))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - width/2, original_means, width, label='Original')
bars2 = ax.bar(x + width/2, mutated_means, width, label='Mutated')

# Labeling
ax.set_ylabel('Mean Score')
ax.set_title('Mean Toxicity, Severe Toxicity, and Insult Scores Comparison')
ax.set_xticks(x)
ax.set_xticklabels(["Toxicity", "Severe Toxicity", "Insult"])
ax.legend()
ax.grid(True, axis='y')
ax.set_facecolor("white")

# Add value labels on top of bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # offset
                    textcoords="offset points",
                    ha='center', va='bottom')

plt.tight_layout()
plt.show()
