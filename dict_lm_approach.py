import os
import sys

import numpy as np
import pandas as pd
import torch
from transformers import BertTokenizer, BertForMaskedLM

from spacy.lang.es import Spanish

AMBIGUITY_TOKEN = "[AMBIGUITY]"


def validate(l_predicted, l_expected):
    num_pos = 0
    num_neg = 0

    for predicted, expected in zip(l_predicted, l_expected):
        predicted = np.array(predicted)
        expected = np.array(expected)

        if predicted.shape != expected.shape or predicted.size == 0:
            raise ValueError(f"Incorrect input arrays {predicted} and {expected}")

        comparison = (predicted == expected)

        pos = np.count_nonzero(comparison)
        neg = len(comparison) - pos

        num_pos += pos
        num_neg += neg

    return num_pos / (num_pos + num_neg)


def get_score(tokens, tokenizer, model, measure="loss"):
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    tensor_input = torch.tensor([token_ids])

    with torch.no_grad():
        outputs = model(tensor_input, labels=tensor_input)
        loss, logit = outputs[:2]

    if measure == "loss":
        result = loss
    elif measure == "loss_times_len":
        result = loss * len(tokens)
    else:
        raise ValueError(f"Incorrect measure {measure}")

    return result


def execute_translation(data_folder, source_lang, target_lang, num_samples, measure, lm_model_name, homographs_file):
    l_predicted = []
    l_expected = []

    # Load models
    lm_tokenizer = BertTokenizer.from_pretrained(lm_model_name)
    lm_model = BertForMaskedLM.from_pretrained(lm_model_name)
    homographs = pd.read_csv(homographs_file)
    nlp = Spanish()
    spacy_tokenizer = nlp.Defaults.create_tokenizer(nlp)

    with open(os.path.join(data_folder, source_lang), "r", encoding="utf-8") as f_input:
        with open(os.path.join(data_folder, target_lang), "r", encoding="utf-8") as f_target:
            for i, (input_text, target_text) in enumerate(zip(f_input.readlines(), f_target.readlines())):
                if 0 < num_samples < i:
                    break

                input_text = input_text.strip().lower()
                target_text = target_text.strip().lower()

                print(f"{i}:")
                print(f"    Input: {input_text}")
                print(f"    Target: {target_text}")

                input_tokens = [token.text for token in spacy_tokenizer(input_text)]
                target_tokens = [token.text for token in spacy_tokenizer(target_text)]

                # First step, fill all non-homographs words
                output_tokens = []
                is_there_amb = False
                for word in input_tokens:
                    if word not in homographs["and"].values:
                        trans_word = word
                    else:
                        word_homographs = eval(homographs[homographs["and"] == word].cas.iloc[0])

                        if len(word_homographs) == 1:
                            trans_word = word_homographs.pop()
                        else:
                            trans_word = AMBIGUITY_TOKEN
                            is_there_amb = True

                    output_tokens.append(trans_word)

                if is_there_amb:  # Here, we need to disambiguate
                    for idx, trans_word in enumerate(output_tokens):
                        if trans_word == AMBIGUITY_TOKEN:
                            candidate_homographs = eval(homographs[homographs["and"] == input_tokens[idx]].cas.iloc[0])
                            best_candidate = None
                            best_score = sys.maxsize
                            for candidate_homograph in candidate_homographs:
                                candidate_output_tokens = output_tokens.copy()
                                candidate_output_tokens[idx] = candidate_homograph
                                candidate_score = get_score(candidate_output_tokens, lm_tokenizer, lm_model, measure)

                                if candidate_score < best_score:
                                    best_score = candidate_score
                                    best_candidate = candidate_homograph

                            if best_candidate is not None:
                                output_tokens[idx] = best_candidate
                            else:
                                output_tokens[idx] = input_tokens[idx]

                l_predicted.append(output_tokens)
                l_expected.append(target_tokens)

                print(f"    Target: {output_tokens}")

    return l_predicted, l_expected


if __name__ == '__main__':
    data_folder = "europarl_unified"
    source_lang = "sentences_and"
    target_lang = "sentences_es"

    num_samples = 100
    measure = "loss"

    lm_model_name = "dccuchile/uncased"  # "dccuchile/cased"

    # Load data
    homographs_file = "homographs.csv"

    l_predicted, l_expected = execute_translation(data_folder, source_lang, target_lang, num_samples, measure,
                                                  lm_model_name, homographs_file)

    print(validate(l_predicted, l_expected))
