

import pandas as pd
import numpy as np

# Set prediction set size (S)
prediction_set_size = 10

# Load the dataset without column names ('new' customer training set or regular training set)
y_customer_df = pd.read_csv('data/y_training.csv', header=None, names=['customer_id', 'basket_id', 'product_id'])

# Load product information 
product_info_df = pd.read_csv('data/products_segmentation.csv')

# Function for the Marginal Probability Recommender
def generate_popularity_recommendations_all(output_path, num_recommendations=prediction_set_size):

    # Get unique customer ids 
    unique_customer_ids = y_customer_df['customer_id'].unique()

    # Calculate the purchase frequency of each product across the entire dataset
    product_purchase_counts = y_customer_df['product_id'].value_counts()
    product_purchase_probs = product_purchase_counts / product_purchase_counts.sum()
    
    # Get the product IDs with the highest purchase probabilities
    top_product_indices = product_purchase_probs.argsort()[::-1][:num_recommendations]
    top_product_ids = product_purchase_probs.index[top_product_indices]
    
    # Initialize an empty list to store recommendations
    recommendations = []

    # Loop through each customer
    for customer_id in unique_customer_ids:

        # Get all available product IDs for the current customer
        customer_product_ids = y_customer_df[y_customer_df['customer_id'] == customer_id]['product_id'].unique()
        
        # Select the product IDs with the highest overall purchase probabilities for recommendations
        recommended_product_ids = np.intersect1d(top_product_ids, customer_product_ids)[:num_recommendations]
        
        # If the number of recommended products is less than num_recommendations, fill with random products
        if len(recommended_product_ids) < num_recommendations:
            available_product_ids = np.setdiff1d(top_product_ids, recommended_product_ids)
            additional_product_ids = np.random.choice(available_product_ids, size=num_recommendations - len(recommended_product_ids), replace=False)
            recommended_product_ids = np.concatenate((recommended_product_ids, additional_product_ids))
        
        # Append recommendations to the list
        for product_id in recommended_product_ids:
            recommendations.append([customer_id, product_info_df[product_info_df['product_id'] == product_id]['lowest_level'].values[0], product_id])

    # Create a DataFrame from the recommendations list
    recommendations_df = pd.DataFrame(recommendations)

    # Save recommendations to a CSV file without column names
    recommendations_df.to_csv(output_path, index=False, header=False)

# Function for the Popularity Based Recommender 
def generate_popularity_recommendations_individual(output_path, num_recommendations=prediction_set_size):
    
    # Get unique customer IDs
    unique_customer_ids = y_customer_df['customer_id'].unique()

    # Initialize an empty list to store recommendations
    recommendations = []

    # Loop through each customer
    for customer_id in unique_customer_ids:

        # Get all available product IDs for the current customer
        customer_product_ids = y_customer_df[y_customer_df['customer_id'] == customer_id]['product_id'].unique()

        # Calculate the purchase frequency of each product for the current customer
        customer_product_purchase_counts = y_customer_df[y_customer_df['customer_id'] == customer_id]['product_id'].value_counts()
        customer_product_purchase_probs = customer_product_purchase_counts / customer_product_purchase_counts.sum()
    
        # Generate recommendations based on purchase frequency probabilities for the current customer
        top_product_indices = np.argsort(customer_product_purchase_probs)[-num_recommendations:]

        # Recommend the most bought products for each customer
        freq_recommendations = customer_product_ids[top_product_indices]

        # Append recommendations to the list
        for product_id in freq_recommendations:
            recommendations.append([customer_id, product_info_df[product_info_df['product_id'] == product_id]['lowest_level'].values[0], product_id])

    # Create a DataFrame from the recommendations list
    recommendations_df = pd.DataFrame(recommendations)

    # Save recommendations to a CSV file without column names
    recommendations_df.to_csv(output_path, index=False, header=False)


# Generate Marginal Probability recommendations  
generate_popularity_recommendations_all('data/prediction_set_marginal.csv')

# Genarate Popularity Based recommendations 
generate_popularity_recommendations_individual('data/prediction_set_popularity.csv')
