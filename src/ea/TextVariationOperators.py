## @file TextVariationOperators.py
# @author Onkar Shelar
#  @brief Contains concrete mutation operators for prompt-level variations.
#
#  This module defines various prompt mutation operators like synonym replacement,
#  deletion, paraphrasing, masked language modeling, and back-translation.
#  All operators implement the VariationOperator interface.
import random
import torch
import spacy
from nltk.corpus import wordnet as wn
from typing import List
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    BertTokenizer,
    BertForMaskedLM,
)
from VariationOperators import VariationOperator
from dotenv import load_dotenv
load_dotenv()

# Load spaCy model once
nlp = spacy.load("en_core_web_sm")

SYNONYMS = {
    "good": ["great", "excellent", "nice"],
    "bad": ["terrible", "awful", "poor"],
    "people": ["individuals", "humans", "persons"],
    "problem": ["issue", "challenge", "difficulty"],
    "happy": ["joyful", "content", "cheerful"],
    "sad": ["unhappy", "miserable", "gloomy"],
}

class SynonymReplacementOperator(VariationOperator):
    def __init__(self):
        super().__init__("SynonymReplacement", "mutation", "Replaces a word with a synonym from a simple dictionary.")

    def apply(self, text: str) -> str:
        words = text.split()
        candidates = [i for i, w in enumerate(words) if w.lower() in SYNONYMS]
        if not candidates:
            return text
        idx = random.choice(candidates)
        word = words[idx].lower()
        words[idx] = random.choice(SYNONYMS[word])
        return " ".join(words)

class RandomDeletionOperator(VariationOperator):
    def __init__(self):
        super().__init__("RandomDeletion", "mutation", "Deletes a random word.")

    def apply(self, text: str) -> str:
        words = text.split()
        if len(words) <= 1:
            return text
        del words[random.randint(0, len(words) - 1)]
        return " ".join(words)

class WordShuffleOperator(VariationOperator):
    def __init__(self):
        super().__init__("WordShuffle", "mutation", "Swaps two adjacent words.")

    def apply(self, text: str) -> str:
        words = text.split()
        if len(words) < 2:
            return text
        idx = random.randint(0, len(words) - 2)
        words[idx], words[idx + 1] = words[idx + 1], words[idx]
        return " ".join(words)

class CharacterSwapOperator(VariationOperator):
    def __init__(self):
        super().__init__("CharacterSwap", "mutation", "Swaps two characters in a random word.")

    def apply(self, text: str) -> str:
        words = text.split()
        idx = random.randint(0, len(words) - 1)
        word = words[idx]
        if len(word) < 2:
            return text
        chars = list(word)
        j = random.randint(0, len(chars) - 2)
        chars[j], chars[j + 1] = chars[j + 1], chars[j]
        words[idx] = "".join(chars)
        return " ".join(words)

class POSAwareSynonymReplacement(VariationOperator):
    def __init__(self):
        super().__init__("POSAwareSynonymReplacement", "mutation", "WordNet synonym replacement based on spaCy POS.")

    def apply(self, text: str) -> str:
        for _ in range(3):
            doc = nlp(text)
            words = text.split()
            if len(words) != len(doc):
                words = [t.text for t in doc]  # realign

            replacements = []
            for token in doc:
                if token.pos_ in {"ADJ", "VERB", "NOUN", "ADV"}:
                    wn_pos = {"ADJ": wn.ADJ, "VERB": wn.VERB, "NOUN": wn.NOUN, "ADV": wn.ADV}[token.pos_]
                    synonyms = {
                        lemma.name().replace("_", " ")
                        for syn in wn.synsets(token.text, pos=wn_pos)
                        for lemma in syn.lemmas()
                        if lemma.name().lower() != token.text.lower()
                    }
                    if synonyms:
                        replacements.append((token.i, random.choice(list(synonyms))))

            if replacements:
                i, replacement = random.choice(replacements)
                if i < len(words):
                    mutated = words.copy()
                    mutated[i] = replacement
                    result = " ".join(mutated)
                    if result.lower().strip() != text.lower().strip():
                        return result

        return text  # fallback

class BertMLMOperator(VariationOperator):
    def __init__(self):
        super().__init__("BertMLM", "mutation", "Uses BERT MLM to replace one word.")
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.model = BertForMaskedLM.from_pretrained("bert-base-uncased")

    def apply(self, text: str) -> str:
        seen = set()
        for _ in range(5):
            words = text.split()
            if not words:
                return text

            idx = random.randint(0, len(words) - 1)
            original = words[idx]
            words[idx] = "[MASK]"
            masked_text = " ".join(words)

            inputs = self.tokenizer(masked_text, return_tensors="pt")
            with torch.no_grad():
                logits = self.model(**inputs).logits

            mask_idx = torch.where(inputs["input_ids"] == self.tokenizer.mask_token_id)[1]
            topk = torch.topk(logits[0, mask_idx], k=5, dim=-1).indices
            sampled = random.choice(topk[0]).item()
            new_word = self.tokenizer.decode([sampled])

            words[idx] = new_word
            result = " ".join(words).strip()
            if result and result.lower() != text.strip().lower() and result.lower() not in seen:
                return result
            seen.add(result.lower())
        return text 

class TinyT5ParaphrasingOperator(VariationOperator):
    def __init__(self):
        super().__init__("TinyT5Paraphrasing", "mutation", "Uses T5 to paraphrase entire input.")
        self.tokenizer = AutoTokenizer.from_pretrained("ramsrigouthamg/t5_paraphraser")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("ramsrigouthamg/t5_paraphraser")
    
    def apply(self, text: str) -> str:
        seen = set()
        for _ in range(3):
            input_text = f"paraphrase: {text} </s>"
            inputs = self.tokenizer.encode_plus(input_text, return_tensors="pt", truncation=True)
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_length=1024,
                    do_sample=True,
                    top_k=50,
                    num_return_sequences=3,
                    early_stopping=True
                )
            for output in outputs:
                result = self.tokenizer.decode(output, skip_special_tokens=True)
                normalized = result.strip().lower()
                if normalized != text.strip().lower() and normalized not in seen:
                    return result
                seen.add(normalized)
        return text

class BackTranslationOperator(VariationOperator):
    def __init__(self):
        super().__init__("BackTranslation", "mutation", "Performs EN→HI→EN back-translation.")
        self.en_hi = pipeline("translation_en_to_hi", model="Helsinki-NLP/opus-mt-en-hi")
        self.hi_en = pipeline("translation_hi_to_en", model="Helsinki-NLP/opus-mt-hi-en")
    
    def apply(self, text: str) -> str:
        seen = set()
        for _ in range(5):
            try:
                hindi = self.en_hi(text, max_length=1024)[0]['translation_text']
                english = self.hi_en(hindi, max_length=1024, do_sample=True, top_k=50)[0]['translation_text']
                if english and english.strip().lower() != text.strip().lower() and english.strip().lower() not in seen:
                    return english
                seen.add(english.strip().lower())
            except Exception as e:
                print(f"[BackTranslation error]: {e}")
                continue
        return text

SINGLE_PARENT_OPERATORS = [
    POSAwareSynonymReplacement(),
    BertMLMOperator(),
    TinyT5ParaphrasingOperator(),
    BackTranslationOperator()
]

MULTI_PARENT_OPERATORS = SINGLE_PARENT_OPERATORS  # can extend with crossover operators later

## @brief Returns the appropriate list of operators depending on the number of parents.
#  @param num_parents Number of parent genomes provided.
#  @return List of applicable variation operators.
def get_applicable_operators(num_parents: int):
    return SINGLE_PARENT_OPERATORS if num_parents == 1 else MULTI_PARENT_OPERATORS