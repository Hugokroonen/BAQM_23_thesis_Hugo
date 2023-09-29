"""
Description:
    In this script, we use the counts_phi data (phi) and the motivation-product probabilities (theta) we have already estimated
    to generate product recommendations. Without using the customer-specific effects (restricted CTM model).
"""

import numpy as np
import pandas as pd
from multiprocessing import Pool

# Set prediction set size (S)
prediction_set_size = 5

# Read motivation-product vectors (phi) 
raw_phi_data = pd.read_csv("output/M30/CTM/counts_phi.csv", header=None) 

# Ensure that the phi values are normalised 
phi_data = raw_phi_data.div(raw_phi_data.sum(axis=0), axis=1)

# Load product information data 
sample_fy_data = pd.read_csv("data/products_segmentation.csv")

# Create a dictionary to map product IDs to lowest levels
product_to_lowest_level_dict = dict(zip(sample_fy_data['product_id'], sample_fy_data['lowest_level']))

# Assign motivation labels based on the column number
motivation_labels = [f"Motivation {i}" for i in range(phi_data.shape[1])]

# Load purchase history of the to predict customers (NEW customers)
# y_data = pd.read_csv("data/y_training_new_customers.csv", header=None)

# Load purchase history of the to predict customers (regular customers)
y_data = pd.read_csv("data/y_training.csv", header=None)

# Assign motivation labels based on the column number
motivation_labels = [f"Motivation {i}" for i in range(phi_data.shape[1])]
motivation_product_vectors = phi_data.values

N, M = motivation_product_vectors.shape  # Number of motivations M and number of products N

# Get number of customers
num_customers = y_data[0].nunique()

# Load motivation-probabilities based on product purchases 
motivation_probabilities = pd.read_csv("output/M30/CTM/customer_motivation_probabilities.csv")
motivation_probabilities = motivation_probabilities.iloc[:, 2:]

# Function to calculate the conditional probabilities (for each product!) given the theta vector
def calculate_conditional_probability(theta_im, motivation_product_vectors):

    # Assuming you have a list of motivation-product vectors for each motivation m
    conditional_probabilities = []
    for m in range(N):
        prob_product_given_motivation = motivation_product_vectors[m]
        # theta @ phi to calculate the conditional probabilities
        conditional_prob = np.dot(theta_im, prob_product_given_motivation)
        conditional_probabilities.append(conditional_prob)

    return conditional_probabilities

def process_customer_LDA(i):
    customer_data = y_data[y_data[0] == i]
    customer_products = len(customer_data[2].values)

    # Select the i'th customer motivation probability vector 
    motivation_probabilities_customer = motivation_probabilities.iloc[i, :]

    # Calculate the conditional probabilities based on theta @ phi (dot product)
    conditional_probabilities = calculate_conditional_probability(motivation_probabilities_customer, motivation_product_vectors)

    # Find the most important motivations for each customer 
    top_motivations_indices = np.argsort(motivation_probabilities_customer)[::-1][:3]
    top_motivations_labels = [motivation_labels[idx] for idx in top_motivations_indices]
    top_motivations_counts = [motivation_probabilities_customer[idx] for idx in top_motivations_indices]

    # Sample X / Select top X products for recommendation based on the highest conditional probabilities
    recommended_product_indices = np.argsort(conditional_probabilities)[::-1][:prediction_set_size]
    recommended_product_ids = [idx for idx in recommended_product_indices]

    recommended_lowest_levels = [product_to_lowest_level_dict.get(product_id, "Unknown") for product_id in recommended_product_ids]
    return i, top_motivations_labels, top_motivations_counts, recommended_lowest_levels

# Generate predictions for the entire set of customers 
if __name__ == '__main__':
    num_workers = 8  # Adjust the number of parallel workers as needed
    with Pool(num_workers) as pool:
        selected_customer_indices = range(num_customers)  # All customers 
        results_LDA = pool.map(process_customer_LDA, selected_customer_indices)

    # Write results to a CSV file for the LDA model
    prediction_data_LDA = []
    for i, _, _, recommended_lowest_levels in results_LDA:
        for product_level in recommended_lowest_levels:
            product_id = None
            for product_id_, product_name in zip(sample_fy_data['product_id'], sample_fy_data['lowest_level']):
                if product_level == product_name:
                    product_id = product_id_
                    break
            prediction_data_LDA.append([i, product_level, product_id])

    prediction_df_LDA = pd.DataFrame(prediction_data_LDA, columns=['customer_id', 'recommended_product_name', 'recommended_product_id'])
    prediction_df_LDA.to_csv('data/prediction_set_restricted_CTM_30.csv', index=False, header=False)  # Set header=False to exclude column names

    for i, top_motivations_labels, top_motivations_counts, recommended_lowest_levels in results_LDA:
        print(f"Customer {i} Top Motivations:")
        for label, count in zip(top_motivations_labels, top_motivations_counts):
            print(f"Motivation: {label} - Probability: {count}")
        print(f"Recommended products for customer {i} (LDA model):")
        for product_level in recommended_lowest_levels:
            print(f"Product: {product_level}")
        print()
