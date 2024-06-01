import pickle
from src.utils.extract_utils import get_mean_head_activations, compute_universal_function_vector
import os, re, json
import torch

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




def find_orthogonal_vector(v1, v2):
    # Ensure vectors are in float format for precision
    v1, v2 = v1.squeeze(), v2.squeeze()
    print("v1: ", v1)
    print("v2: ", v2)
    print("v1.size(): ", v1.size())
    print("v2.size(): ", v2.size())
    v1 = v1.float()
    v2 = v2.float()

    # Normalize v1 to avoid growing the values too large in dot products
    v1 = v1 / torch.norm(v1)

    # Project v2 onto v1 and subtract to get part of v2 orthogonal to v1
    projection_v2_on_v1 = torch.dot(v2, v1) * v1
    v2_orthogonal_to_v1 = v2 - projection_v2_on_v1

    # Normalize the second vector after making it orthogonal to the first
    v2_orthogonal_to_v1 = v2_orthogonal_to_v1 / torch.norm(v2_orthogonal_to_v1)

    # Generate a random vector
    random_vector = torch.randn_like(v1)

    # Make the random vector orthogonal to both v1 and the adjusted v2
    projection_random_on_v1 = torch.dot(random_vector, v1) * v1
    projection_random_on_v2 = torch.dot(random_vector, v2_orthogonal_to_v1) * v2_orthogonal_to_v1
    orthogonal_vector = random_vector - projection_random_on_v1 - projection_random_on_v2

    # Normalize the final orthogonal vector
    orthogonal_vector = orthogonal_vector / torch.norm(orthogonal_vector)

    return orthogonal_vector