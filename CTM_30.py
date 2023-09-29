"""
Description:
    In this script, we use the counts_phi data (phi), the motivation-product probabilities (theta) and the customer-specific
    effects (alpha) we have already estimated to generate product recommendations (full CTM model). 
"""

import numpy as np
import pandas as pd
from multiprocessing import Pool

# Set prediction set size (S)
prediction_set_size = 3

# Load the motivation-product vectors (phi) - that are already estimated 
raw_phi_data = pd.read_csv("output/M30/CTM/counts_phi.csv", header=None) 

# Ensure that the phi values are normalised 
phi_data = raw_phi_data.div(raw_phi_data.sum(axis=0), axis=1)

# Load customer-specific data of 'to predict' customers 
wi_data = pd.read_csv("data/h_segmentation.csv", header=None)  

# Load product information data
sample_fy_data = pd.read_csv("data/products_segmentation.csv")

# Create a dictionary to map product IDs to lowest levels
product_to_lowest_level = dict(zip(sample_fy_data['product_id'], sample_fy_data['lowest_level']))

# Assign motivation labels based on the column number
motivation_labels = [f"Motivation {i}" for i in range(phi_data.shape[1])]

# Load purchase history of the to predict customers (regular customers)
y_data = pd.read_csv("data/y_training.csv", header=None)

# Load purchase history of the to predict customers (NEW customers)
# y_data = pd.read_csv("data/y_training_new_customers.csv", header=None)

# Load gamma and kappa data
gamma_data = pd.read_csv("output/M30/CTM/gamma.csv")
kappa_data = pd.read_csv("output/M30/CTM/kappa.csv")

# Assign motivation labels based on the column number
motivation_labels = [f"Motivation {i}" for i in range(phi_data.shape[1])]
motivation_product_vectors = phi_data.values

N, M = motivation_product_vectors.shape  # Number of motivations M and number of products N

# Get total number of customers to predict 
num_customers = y_data[0].nunique()

# Load the estimated motivation probabilities for each customer (segmentation results)
motivation_probabilities = pd.read_csv("output/M30/CTM/customer_motivation_probabilities.csv")
motivation_probabilities = motivation_probabilities.iloc[:, 2:]

# Function to calculate the mean of the theta vector for each customer
# If more products are bought, the theta mean will approach the motivation probabilities, hence alpha having less effect
def calculate_mean_theta(alpha_im, motivation_probabilities):
    mean_theta_im = np.zeros(len(alpha_im))
    for m in range(len(alpha_im)):
        # Combining customer specific effects and motivation probabilities
        mean_theta_im[m] = (alpha_im[m] + motivation_probabilities[m]) / (np.sum(alpha_im) + np.sum(motivation_probabilities))

    return mean_theta_im

# Function to calculate the conditional probabilities (for each product!) given the theta vector
def calculate_conditional_probability(theta_im, motivation_product_vectors):
    
    # Phi @ Theta 
    conditional_probabilities = motivation_product_vectors @ theta_im 

    return conditional_probabilities

# Function to generate recommendations for a customer
def process_customer(i):
    customer_data = y_data[y_data[0] == i]
    customer_products = len(customer_data[2].values)
    # Possibly introduce a weight if alpha has too much effect 
    # Alpha_weight = 1 / (1 + 0.05 * customer_products) 
    alpha_weight = 1 # No weight 

    # Calculate the alpha values for each customer (vector over all motivations) 
    alpha_vector = alpha_weight * np.exp(kappa_data.mean() + np.dot(wi_data.values[i], gamma_data.T))

    # Select the i'th customer motivation probability vector 
    motivation_probabilities_customer = motivation_probabilities.iloc[i, :]

    # Calculate the theta based on product purchases and customer-specific information 
    theta_mean = calculate_mean_theta(alpha_vector, motivation_probabilities_customer)


    # Phi @ Theta results in conditional probabilities for all products 
    conditional_probabilities = calculate_conditional_probability(theta_mean, motivation_product_vectors)

    # Select the products with the highest probability for recommendation 
    top_motivations_indices = np.argsort(motivation_probabilities_customer)[::-1][:3]
    top_motivations_labels = [motivation_labels[idx] for idx in top_motivations_indices]
    top_motivations_counts = [motivation_probabilities_customer[idx] for idx in top_motivations_indices]

    # Sample X products  / Select highest probable products for recommendation based on conditional probabilities
    # recommended_product_indices = np.random.choice(len(conditional_probabilities), size=3, p=conditional_probabilities)
    recommended_product_indices = np.argsort(conditional_probabilities)[::-1][:prediction_set_size]
    recommended_product_ids = [idx for idx in recommended_product_indices]

    recommended_lowest_levels = [product_to_lowest_level.get(product_id, "Unknown") for product_id in recommended_product_ids]
    return i, top_motivations_labels, top_motivations_counts, recommended_lowest_levels

# Generate product recommendations for the entire set of customers 
if __name__ == '__main__':
    num_workers = 8  # Adjust the number of parallel workers as needed
    with Pool(num_workers) as pool:
        selected_customer_indices = range(num_customers)  # All customers 
        results = pool.map(process_customer, selected_customer_indices)

    # Write results to a CSV file
    prediction_data = []
    for i, _, _, recommended_lowest_levels in results:
        for product_level in recommended_lowest_levels:
            product_id = None
            for product_id_, product_name in zip(sample_fy_data['product_id'], sample_fy_data['lowest_level']):
                if product_level == product_name:
                    product_id = product_id_
                    break
            prediction_data.append([i, product_level, product_id])

    prediction_df = pd.DataFrame(prediction_data, columns=['customer_id', 'recommended_product_name', 'recommended_product_id'])
    prediction_df.to_csv('data/prediction_set_CTM_30.csv', index=False, header=False)  # Set header=False to exclude column names

    # Print the recommendations for each customer 
    for i, top_motivations_labels, top_motivations_counts, recommended_lowest_levels in results:
        print(f"Customer {i} Top Motivations:")
        for label, count in zip(top_motivations_labels, top_motivations_counts):
            print(f"Motivation: {label} - Probability: {count}")
        print(f"Recommended products for customer {i}:")
        for product_level in recommended_lowest_levels:
            print(f"Product: {product_level}")
        print()
