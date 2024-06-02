import pandas as pd
# import matplotlib.pyplot as plt
from transformers import pipeline
from numpy.linalg import norm
import random
import pickle
from tqdm import tqdm
random.seed(10)

import os, re, json
import torch, numpy as np
from omegaconf import DictConfig, OmegaConf

import sys
# sys.path.append('../')
torch.set_grad_enabled(False)

from src.utils.extract_utils import get_mean_head_activations, compute_universal_function_vector
from src.utils.intervention_utils import fv_intervention_natural_text, function_vector_intervention
from src.utils.model_utils import load_gpt_model_and_tokenizer
from src.utils.prompt_utils import load_dataset, word_pairs_to_prompt_data, create_prompt
from src.utils.eval_utils import decode_to_vocab, sentence_eval, get_probability, is_highest_probability, is_highest_probability_and_sorted_indices
from utils import compute_and_cache_FV, get_bias, get_score, avg, find_orthogonal_vector
import sys
import sys
import argparse
import json
import ast
import json

parser = argparse.ArgumentParser()
parser.add_argument('--cfg', default='default.cfg', help='Configuration file')
args = parser.parse_args()

config = args.cfg
config = OmegaConf.create(config)


model_name = config['model']['name']
model, tokenizer, model_config = load_gpt_model_and_tokenizer(model_name)


# Import the variable from the file
if config['eval']['classification_labels']:
    with open(f"./dataset_files/labels/{config['eval']['classification_labels']}.json", 'r') as f:
        classification_labels = json.load(f)['labels']
print("classification_labels: ", classification_labels)
print("classification_labels: ", classification_labels)
# Load Datasets
fv_dataset_name = config['dataset']['fv_dataset']
fv_dataset = load_dataset(fv_dataset_name, seed=0)
fv_intervention_datasets_names = config['dataset']['fv_intervention_dataset']
classification_dataset = config['eval']['classification_dataset']
intervention_datasets = {}

for dataset_name in fv_intervention_datasets_names:
    dataset = load_dataset(dataset_name, seed=0)
    intervention_datasets[dataset_name] = dataset
    
profession_dataset = load_dataset(classification_dataset, seed=0)

# Build Function Vectors from datasets
FV, top_heads = compute_and_cache_FV(fv_dataset, model, model_config, tokenizer, cache_filename=f"FVs/FV_{fv_dataset_name}.pkl")

FV_intervention_list = []
top_heads_intervention_list = []
for intervention_dataset in intervention_datasets:
    FV_intervention, top_heads_intervention = compute_and_cache_FV(intervention_datasets[intervention_dataset], model, model_config, tokenizer, cache_filename=f"FVs/FV_intervention_{intervention_dataset}.pkl")
    FV_intervention_list.append(FV_intervention)
    top_heads_intervention_list.append(top_heads_intervention)
    
# FV_female, top_heads_female = compute_and_cache_FV(female_dataset, model, model_config, tokenizer, cache_filename='FVs/FV_female.pkl')
# FV_male, top_heads_male = compute_and_cache_FV(male_dataset, model, model_config, tokenizer, cache_filename='FVs/FV_male.pkl')

# FV_profession, top_heads_profession = compute_and_cache_FV(profession_dataset, model, model_config, tokenizer, cache_filename=f"FVs/FV_classification{classification_dataset}.pkl")

# FV_orthogonal = find_orthogonal_vector(FV_male, FV_female)


# print("FV_orthogonal: ", FV_orthogonal)
# FV  = torch.randn(FV.size()).to(FV_female.device)
# FV = FV_random
total_prob_gender_zero_shot_bias_male = []
total_prob_gender_zero_shot_bias_female = []
total_prob_gender_icl_bias_male = []
total_prob_gender_icl_bias_female = []
total_prob_gender_fv_bias_male = []
total_prob_gender_fv_bias_female = []
total_prob_icl_fv_bias_male = []
total_prob_icl_fv_bias_female = []
total_prob_gender_icl_fv_minus_fvunb_bias_male = []
total_prob_gender_icl_fv_minus_fvunb_bias_female = []
total_prob_gender_icl_fv_plus_fvunb_bias_male = []
total_prob_gender_icl_fv_plus_fvunb_bias_female = []
total_prob_gender_fv_unbias_male = []
total_prob_gender_fv_unbias_female = []

total_prob_gender_fv_plus_fvunb_profession = []
total_prob_gender_zero_shot_plus_fvunb_profession = []

total_prob_gender_fv_plus_fvunb_profession_classify = []
total_prob_gender_zero_shot_plus_fvunb_profession_classify=[]
total_prob_gender_zero_shot_plus_fvunb_profession_classify_all = {}
total_prob_gender_fv_plus_fvunb_profession_classify_all = {}
for n_top_heads in range(10,11):

    # Sample ICL example pairs, and a test word
    word_pairs = fv_dataset['train'][:10]
    word_pairs_profession = profession_dataset['train'][:10]
    
    for test_pair_idx in tqdm(range(15, 100)):
        
        # test-set
        test_pair = fv_dataset['test'][test_pair_idx]
        test_profession = profession_dataset['test'][test_pair_idx]
        test_profession['output'] = str(test_profession['output'])
        
        # Prompts
        prompt_data = word_pairs_to_prompt_data(word_pairs, query_target_pair=test_pair, prepend_bos_token=True)
        prompt_data_profession = word_pairs_to_prompt_data(word_pairs_profession, query_target_pair=test_profession, prepend_bos_token=True)
        zeroshot_prompt_data = word_pairs_to_prompt_data({'input':[], 'output':[]}, query_target_pair=test_pair, prepend_bos_token=True, shuffle_labels=True)
        zeroshot_prompt_data_profession = word_pairs_to_prompt_data({'input':[], 'output':[]}, query_target_pair=test_profession, prepend_bos_token=True, shuffle_labels=True)
        
        sentence = create_prompt(prompt_data)
        sentence_profession = create_prompt(prompt_data_profession)
        zeroshot_sentence = create_prompt(zeroshot_prompt_data)
        zeroshot_sentence_profession = create_prompt(zeroshot_prompt_data_profession)

        # No FV Logits
        clean_logits = sentence_eval(sentence, [test_pair['output']], model, tokenizer, compute_nll=False)
        zero_shot_logits = sentence_eval(zeroshot_sentence, [test_pair['output']], model, tokenizer, compute_nll=False)
        zero_shot_logits_profession = sentence_eval(zeroshot_sentence_profession, [test_profession['output']], model, tokenizer, compute_nll=False)

        # Intervention on the zero-shot prompt at layer i. Layer 21 was found to be the most optimal in the paper
        for i in range(21, 22):
            
            # Use FV as intervention on LLM query
            _, interv_logits_fv = function_vector_intervention(zeroshot_sentence, [test_pair['output']], i, FV, model, model_config, tokenizer)
            _, interv_logits_icl = function_vector_intervention(sentence, [test_pair['output']], i, FV, model, model_config, tokenizer)
            _, interv_logits_both_minus = function_vector_intervention(sentence, [test_pair['output']], i, FV- 1 * (sum(FV_intervention_list)), model, model_config, tokenizer)
            _, interv_logits_both_plus = function_vector_intervention(sentence, [test_pair['output']], i, FV+ 1 * (sum(FV_intervention_list)), model, model_config, tokenizer)
            _, interv_logits_fvunb = function_vector_intervention(sentence, [test_pair['output']], i, 1 * (sum(FV_intervention_list)) , model, model_config, tokenizer)
            _, interv_logits_both_plus_profession = function_vector_intervention(zeroshot_sentence_profession, [test_profession['output']], i, FV+ 1 * (sum(FV_intervention_list)), model, model_config, tokenizer)
            
            
            # Get token probability of output strings "male" and "female" after intervention
            prob_gender_zero_shot = [get_probability(zero_shot_logits, tokenizer, "male"),  get_probability(zero_shot_logits, tokenizer, "female")]
            prob_gender_icl = [get_probability(clean_logits, tokenizer, "male"),  get_probability(clean_logits, tokenizer, "female")]
            prob_gender_fv = [get_probability(interv_logits_fv, tokenizer, "male"), get_probability(interv_logits_fv, tokenizer, "female")]
            prob_gender_icl_fv = [get_probability(interv_logits_icl, tokenizer, "male"), get_probability(interv_logits_icl, tokenizer, "female")]
            prob_gender_icl_fv_minus_fvunb = [get_probability(interv_logits_both_minus, tokenizer, "male"), get_probability(interv_logits_both_minus, tokenizer, "female")]
            prob_gender_icl_fv_plus_fvunb = [get_probability(interv_logits_both_plus, tokenizer, "male"), get_probability(interv_logits_both_plus, tokenizer, "female"),]
            prob_gender_fvunb = [get_probability(interv_logits_fvunb, tokenizer, "male"), get_probability(interv_logits_fvunb, tokenizer, "female"),]
            
            # calculate the normalized probability of gender tokens
            prob_gender_zero_shot_bias_male, prob_gender_zero_shot_bias_female = get_bias(prob_gender_zero_shot)
            prob_gender_icl_bias_male, prob_gender_icl_bias_female = get_bias(prob_gender_icl)
            prob_gender_fv_bias_male, prob_gender_fv_bias_female = get_bias(prob_gender_fv)
            prob_icl_fv_bias_male, prob_icl_fv_bias_female = get_bias(prob_gender_icl_fv)
            prob_gender_icl_fv_minus_fvunb_bias_male, prob_gender_icl_fv_minus_fvunb_bias_female = get_bias(prob_gender_icl_fv_minus_fvunb)
            prob_gender_icl_fv_plus_fvunb_bias_male, prob_gender_icl_fv_plus_fvunb_bias_female = get_bias(prob_gender_icl_fv_plus_fvunb)
            prob_gender_fv_unbias_male, prob_gender_fv_unbias_female = get_bias(prob_gender_fvunb)
            
            # only store male prob since male bias = female bias bc of normalization
            total_prob_gender_zero_shot_bias_male.append(prob_gender_zero_shot_bias_male)
            total_prob_gender_icl_bias_male.append(prob_gender_icl_bias_male)
            total_prob_gender_fv_bias_male.append(prob_gender_fv_bias_male)
            total_prob_icl_fv_bias_male.append(prob_icl_fv_bias_male)
            total_prob_gender_icl_fv_minus_fvunb_bias_male.append(prob_gender_icl_fv_minus_fvunb_bias_male)
            total_prob_gender_icl_fv_plus_fvunb_bias_male.append(prob_gender_icl_fv_plus_fvunb_bias_male)
            total_prob_gender_fv_unbias_male.append(prob_gender_fv_unbias_male)
            
            # Get probability and classification performance of profession prediction
            profession_word = classification_labels[int(test_profession["output"])]
            
            prob_gender_zero_shot_profession = get_probability(zero_shot_logits_profession, tokenizer, profession_word)
            prob_gender_fv_plus_fvunb_profession = get_probability(interv_logits_both_plus_profession, tokenizer, profession_word)
            prob_gender_zero_shot_profession_classify, prob_gender_zero_shot_profession_classify_all = is_highest_probability_and_sorted_indices(zero_shot_logits_profession, tokenizer, profession_word, classification_labels)
            prob_gender_fv_plus_fvunb_profession_classify, prob_gender_fv_plus_fvunb_profession_classify_all = is_highest_probability_and_sorted_indices(interv_logits_both_plus_profession, tokenizer, profession_word, classification_labels)
            # prob_gender_zero_shot_profession_classify = is_highest_probability(zero_shot_logits_profession, tokenizer, profession_word, classification_labels)
            # prob_gender_fv_plus_fvunb_profession_classify = is_highest_probability(interv_logits_both_plus_profession, tokenizer, profession_word, classification_labels)
            
            # accumulate probability and classification performance of profession prediction
            total_prob_gender_zero_shot_plus_fvunb_profession.append(prob_gender_zero_shot_profession)
            total_prob_gender_fv_plus_fvunb_profession.append(prob_gender_fv_plus_fvunb_profession)
            total_prob_gender_zero_shot_plus_fvunb_profession_classify.append(prob_gender_zero_shot_profession_classify)
            total_prob_gender_fv_plus_fvunb_profession_classify.append(prob_gender_fv_plus_fvunb_profession_classify)
            
            for label, probability in prob_gender_zero_shot_profession_classify_all:
                if label not in total_prob_gender_zero_shot_plus_fvunb_profession_classify_all:
                    total_prob_gender_zero_shot_plus_fvunb_profession_classify_all[label] = []
                total_prob_gender_zero_shot_plus_fvunb_profession_classify_all[label].append(probability)

            for label, probability in prob_gender_fv_plus_fvunb_profession_classify_all:
                if label not in total_prob_gender_fv_plus_fvunb_profession_classify_all:
                    total_prob_gender_fv_plus_fvunb_profession_classify_all[label] = []
                total_prob_gender_fv_plus_fvunb_profession_classify_all[label].append(probability)
                
    total_prob_gender_zero_shot_plus_fvunb_profession_classify_all_labels = {}
    for label, probs in total_prob_gender_zero_shot_plus_fvunb_profession_classify_all.items():
        total_prob_gender_zero_shot_plus_fvunb_profession_classify_all[label] = sum(probs) / len(probs)
        # label_name = classification_labels[label]
        # total_prob_gender_zero_shot_plus_fvunb_profession_classify_all_labels[label_name] = sum(probs) / len(probs)

    total_prob_gender_fv_plus_fvunb_profession_classify_all_labels = {}
    for label, probs in total_prob_gender_fv_plus_fvunb_profession_classify_all.items():
        # label_name = classification_labels[label]
        total_prob_gender_fv_plus_fvunb_profession_classify_all[label] = sum(probs) / len(probs)
        # total_prob_gender_fv_plus_fvunb_profession_classify_all_labels[label_name] = sum(probs) / len(probs)

            
    # Gender Bias Calculation
    import matplotlib.pyplot as plt

    # Gender Bias Calculation
    import matplotlib.pyplot as plt

    def plot_results(results_dict):
        # labels = [label.replace("total_prob_", "") for label in results_dict.keys()]
        labels = [label for label in results_dict.keys()]
        values = list(results_dict.values())
        # Plot the results
        plt.bar(labels, values)
        plt.xlabel("Bias Metric")
        plt.ylabel("Average Probability (%)")
        plt.title("Gender Bias Calculation")
        plt.xticks(rotation=45)
        plt.savefig("plots/gender_bias_calculation.png")  # Save the plot as an image
        plt.close()  # Close the plot to free up memory

    # Create a dictionary with the results
    results_dict = {
        "gender_zero_shot_bias_male": avg(total_prob_gender_zero_shot_bias_male) * 100,
        "gender_icl_bias_male": avg(total_prob_gender_icl_bias_male) * 100,
        "gender_fv_bias_male": avg(total_prob_gender_fv_bias_male) * 100,
        "icl_fv_bias_male": avg(total_prob_icl_fv_bias_male) * 100,
        "gender_icl_fv_minus_fvunb_bias_male": avg(total_prob_gender_icl_fv_minus_fvunb_bias_male) * 100,
        "gender_icl_fv_plus_fvunb_bias_male": avg(total_prob_gender_icl_fv_plus_fvunb_bias_male) * 100,
        "gender_fv_unbias_male": avg(total_prob_gender_fv_unbias_male) * 100
    }
    # Plot the results
    plot_results(results_dict)

    # print("total_prob_gender_zero_shot_bias_male: ", avg(total_prob_gender_zero_shot_bias_male) * 100)
    # print("total_prob_gender_icl_bias_male: ", avg(total_prob_gender_icl_bias_male) * 100)
    # print("total_prob_gender_fv_bias_male: ", avg(total_prob_gender_fv_bias_male) * 100)
    # print("total_prob_icl_fv_bias_male: ", avg(total_prob_icl_fv_bias_male) * 100)
    # print("total_prob_gender_icl_fv_minus_fvunb_bias_male: ", avg(total_prob_gender_icl_fv_minus_fvunb_bias_male) * 100)
    # print("total_prob_gender_icl_fv_plus_fvunb_bias_male: ", avg(total_prob_gender_icl_fv_plus_fvunb_bias_male) * 100)
    # print("total_prob_gender_fv_unbias_male: ", avg(total_prob_gender_fv_unbias_male) * 100)
    
    # Profession classification Performance
    
    def plot_profession_results(results_dict):
        labels = [label for label in results_dict.keys()]
        values = list(results_dict.values())
        plt.bar(labels, values)
        plt.xlabel("Profession")
        plt.ylabel("Average Probability (%)")
        plt.title("Profession Classification Performance")
        plt.xticks(rotation=45)
        plt.savefig("plots/profession_classification_performance.png")  # Save the plot as an image
        plt.close()

    plot_profession_results({
        "zero_shot_plus_fvunb_profession": avg(total_prob_gender_zero_shot_plus_fvunb_profession) * 100,
        "fv_plus_fvunb_profession": avg(total_prob_gender_fv_plus_fvunb_profession) * 100,
        "zero_shot_plus_fvunb_profession_classify": avg(total_prob_gender_zero_shot_plus_fvunb_profession_classify) * 100,
        "fv_plus_fvunb_profession_classify": avg(total_prob_gender_fv_plus_fvunb_profession_classify) * 100
    })
    # print("total_prob_gender_zero_shot_plus_fvunb_profession: ", avg(total_prob_gender_zero_shot_plus_fvunb_profession) * 100)
    # print("total_prob_gender_fv_plus_fvunb_profession: ", avg(total_prob_gender_fv_plus_fvunb_profession) * 100)
    # print("total_prob_gender_zero_shot_plus_fvunb_profession_classify: ", avg(total_prob_gender_zero_shot_plus_fvunb_profession_classify) * 100)
    # print("total_prob_gender_fv_plus_fvunb_profession_classify: ", avg(total_prob_gender_fv_plus_fvunb_profession_classify) * 100)
    
    
    def plot_classification_results(results_dict, name):
        labels = [label for label in results_dict.keys()]
        values = list(results_dict.values())
        plt.bar(labels, values)
        plt.xlabel("Profession")
        plt.ylabel("Average Probability (%)")
        plt.title("Profession Classification Performance")
        plt.xticks(rotation=45)
        plt.savefig("plots/profession_classification_performance_all{name}.png")  # Save the plot as an image
        plt.close()
        

    plot_classification_results(total_prob_gender_zero_shot_plus_fvunb_profession_classify_all, "baseline")
    plot_classification_results(total_prob_gender_fv_plus_fvunb_profession_classify_all, "fv")
    # print("total_prob_gender_zero_shot_plus_fvunb_profession_classify_all: ", total_prob_gender_zero_shot_plus_fvunb_profession_classify_all)
    # print("total_prob_gender_fv_plus_fvunb_profession_classify_all: ", total_prob_gender_fv_plus_fvunb_profession_classify_all)
    