import pickle
from src.utils.extract_utils import get_mean_head_activations, compute_universal_function_vector
import os, re, json


def save_to_cache(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def load_from_cache(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)
    
def compute_and_cache_FV(dataset, model, model_config, tokenizer, cache_filename='fv_cache.pkl'):
    if os.path.exists(cache_filename):
        print("Loading FV from cache.")
        FV, top_heads = load_from_cache(cache_filename)
    else:
        print("Computing FV.")
        mean_activations = get_mean_head_activations(dataset, model, model_config, tokenizer)
        FV, top_heads = compute_universal_function_vector(mean_activations, model, model_config, n_top_heads=10)
        save_to_cache((FV, top_heads), cache_filename)
    
    return FV, top_heads
def get_bias(inputs):
    male_bias = inputs[0] / sum(inputs)
    female_bias = inputs[1] / sum(inputs)
    male_bias = abs(.5 - male_bias)
    female_bias = abs(.5 - female_bias)
    
    return male_bias, female_bias
    
def avg(l):
    return sum(l) / len(l)

def get_score(score, test_sample):
    top1=[]
    top5=[]
    top10=[]
    k_found = 100
    ground_truth =  test_sample.strip().lower()
    for k in range(len(score)):
        prediction = score[k][0].strip().lower()
        if prediction == ground_truth:
              k_found = k
    if k_found == 0:
        top1.append(1)
        top5.append(1)
        top10.append(1)
    elif 1 <= k_found <=4:
        top1.append(0)
        top5.append(1)
        top10.append(1)
    elif 1<= k_found <=9:
        top1.append(0)
        top5.append(0)
        top10.append(1)
    else:
        top1.append(0)
        top5.append(0)
        top10.append(0)
    
    top_1_accuracy = sum(top1) / len(top1)
    top_5_accuracy = sum(top5) / len(top5)
    top_10_accuracy = sum(top10) / len(top10)
    return top_1_accuracy, top_5_accuracy, top_10_accuracy