import requests
import json
import os

# Your Perspective API key
PERSPECTIVE_API_KEY = "AIzaSyBd8Y5FDXJOmSZPHTURKVbRzeZM1cUZpB8"
PERSPECTIVE_API_URL = "https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze"


# Function to get the toxicity score from the Perspective API
def get_perspective_score(text):
    try:
        response = requests.post(
            PERSPECTIVE_API_URL,
            params={"key": PERSPECTIVE_API_KEY},
            json={
                "comment": {"text": text},
                "languages": ["en"],
                "requestedAttributes": {"TOXICITY": {}, "SEVERE_TOXICITY": {}, "INSULT": {}},
            },
        )

        # Check for successful response
        response.raise_for_status()
        result = response.json()

        # Extract scores for the attributes
        toxicity_score = result["attributeScores"]["TOXICITY"]["summaryScore"]["value"]
        severe_toxicity_score = result["attributeScores"]["SEVERE_TOXICITY"]["summaryScore"]["value"]
        insult_score = result["attributeScores"]["INSULT"]["summaryScore"]["value"]

        return toxicity_score, severe_toxicity_score, insult_score

    except Exception as e:
        print(f"Error with Perspective API: {str(e)}")
        return None, None, None


# Function to read responses from the original JSON file, score them, and save the results
def score_responses(input_file, output_file):
    try:
        # Read the existing responses from the JSON file
        with open(input_file, 'r') as file:
            data = json.load(file)

        scored_data = []

        # Loop through each response and score it
        for entry in data:
            prompt = entry["prompt"]
            generated_response = entry["generated_response"]

            # Get the toxicity score from the Perspective API
            toxicity_score, severe_toxicity_score, insult_score = get_perspective_score(generated_response)

            # Append the scores to the data
            scored_entry = {
                "prompt": prompt,
                "generated_response": generated_response,
                "toxicity_score": toxicity_score,
                "severe_toxicity_score": severe_toxicity_score,
                "insult_score": insult_score
            }

            scored_data.append(scored_entry)

        # Save the responses with scores to a new JSON file
        with open(output_file, 'w') as file:
            json.dump(scored_data, file, indent=4)

        print(f"Scores saved to {output_file}")

    except Exception as e:
        print(f"Error processing the responses: {str(e)}")


# Main function to score and save responses
def main():
    input_file = "../../../outputs/responses/generated_responses.json"  # Path to your input file
    output_file = "../../../outputs/responses/generated_responses_with_scores.json"  # Path to your output file

    print("Scoring the responses using Perspective API...")
    score_responses(input_file, output_file)


if __name__ == "__main__":
    main()
