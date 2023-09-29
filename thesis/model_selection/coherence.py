"""
Description:
    In this script, we calculate the coherence of each motivation based on the NPMI, as well as the overall
    coherence of all motivations to be used as a model selection criterion. 
"""

import pandas as pd
import numpy as np
from math import log2
from itertools import combinations

# Function to calculate NPMI 
def calculate_npmi(dataset, products):

    # Total amount of baskets 
    total_transactions = len(dataset[1].unique())
    npmi_scores = []

    # All possible pairs of products 
    product_pairs = list(combinations(products, 2))

    # For all product pairs 
    for pair in product_pairs:
     
        # Baskets containing product 1 in sample 
        product_1_baskets = dataset[dataset[2] == pair[0]][1].unique()
        # Baskets containing product 2 in sample 
        product_2_baskets = dataset[dataset[2] == pair[1]][1].unique()

        # Total frequency of baskets with product 1 in sample 
        product_1_freq = len(product_1_baskets)
        # Total frequency of baskets with product 2 in sample 
        product_2_freq = len(product_2_baskets)

        # How many times the two products co-occur in a basket 
        product_pair_freq = len(set(product_1_baskets) & set(product_2_baskets))

        # If a pair never co-occurs in a basket, append the lowest possible NPMI score for that pair 
        if product_pair_freq == 0:
            npmi_scores.append(-1)
        # Otherwise, calculate NPMI 
        else:
            pmi = log2((product_pair_freq / total_transactions) / ((product_1_freq / total_transactions) * (product_2_freq / total_transactions)))
            npmi = pmi / (-log2(product_pair_freq / total_transactions))
            npmi_scores.append(npmi)

    # Calculate the average NPMI 
    average_npmi = np.mean(npmi_scores)
    return average_npmi

# Load the customer dataset
dataset = pd.read_csv('data/y.csv', header=None)

# Read motivations and corresponding products from file (no probabilities needed)
motivations_df = pd.read_csv('output/M30/CTM/motivations_top_products.csv')
motivations = motivations_df.columns[1:]  # Exclude the first column (product IDs)

# To store all average NPMI scores
average_npmi_list = []

# Calculate NPMI per motivation 
for i, motivation in enumerate(motivations):
    products = motivations_df[motivation].tolist()
    average_npmi = calculate_npmi(dataset, products)
    average_npmi_list.append(average_npmi)
    print("NPMI for", motivation, ":", average_npmi)

# Calculate average NPMI across all motivations 
average_npmi_all_motivations = np.mean(average_npmi_list)
print("Average NPMI across all motivations:", average_npmi_all_motivations)
