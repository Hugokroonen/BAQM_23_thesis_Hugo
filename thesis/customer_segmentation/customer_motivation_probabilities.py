import pandas as pd
import numpy as np
import pandas as pd
from multiprocessing import Pool
import time 

# First, align product IDs and lowest level mapping of fitted model and 'to-predict' customer set 
# This is needed for later use in product recommendations, not for customer segmentation 

# Load the product data of the already fitted model CTM model 
final_sample_df = pd.read_csv('data/products.csv')

# Load the product data of the set of 'to predict' customers
final_sample_predict_df = pd.read_csv('data/products_segmentation.csv')

# Create a dictionary to map lowest_level to product_id from fitted model 
lowest_level_to_product = final_sample_df.set_index('lowest_level')['product_id'].to_dict()

# Apply the mapping to 'to-predict' set of customers 
final_sample_predict_df['product_id'] = final_sample_predict_df['lowest_level'].map(lowest_level_to_product)

# Replace products that were not in the fitted model (NaN or inf) with -1 as a placeholder
final_sample_predict_df['product_id'].fillna(-1, inplace=True)
final_sample_predict_df['product_id'].replace([float('inf'), float('-inf')], -1, inplace=True)

# Convert product_id to integers
final_sample_predict_df['product_id'] = final_sample_predict_df['product_id'].astype(int)

# Save the updated product information to a new CSV file to be used for prediction 
final_sample_predict_df.to_csv('data/products_segmentation.csv', index=False)

print("Alignment of product id's complete")

# Load necessary data 

# Motivation-product vectors (phi), already fitted 
phi_data_raw = pd.read_csv("output/M30/CTM/counts_phi.csv", header=None) 

# Ensure that phi data is normalised 
normalized_phi_data = phi_data_raw.div(phi_data_raw.sum(axis=0), axis=1)

# Clip probabilities above 0.10 to 0.10 - to remove outliers that have great effect
phi_data = np.clip(normalized_phi_data, a_min=None, a_max=0.10)

# Assign motivation labels based on the column number
motivation_labels = [f"Motivation {i}" for i in range(phi_data.shape[1])]

# Load customers we are going to segment 
y_data = pd.read_csv('data/y_segmentation.csv', header=None) 

# Replace possible (erroneous) string values with 0 
y_data.iloc[:, 2] = y_data.iloc[:, 2].apply(lambda x: 0 if isinstance(x, str) else x) 

# Get number of customers
num_customers = y_data[0].nunique()

# Processing customers 

# Calculate the purchase frequency vector over distinct products
distinct_product_ids = np.unique(y_data.iloc[:, 2])
num_rows_counts_phi = phi_data_raw.shape[0]
unique_products_counts_phi = list(range(num_rows_counts_phi))

# Function that generates segmentation for a customer  
def process_customer(i):
    customer_data = y_data[y_data[0] == i]
    customer_products = customer_data[2].values
    product_purchase_frequencies = np.zeros(len(unique_products_counts_phi))

    # Generate a purchase frequency vector over the product assortment 
    for product_id in customer_products:
        if product_id in unique_products_counts_phi:
            product_purchase_frequencies[np.where(unique_products_counts_phi == product_id)[0][0]] += 1

    # Calculate the dot product between this vector and phi data to get motivation proportions for this customer 
    motivation_probabilities = np.dot(product_purchase_frequencies, phi_data)
    sum_motivation_probabilities = np.sum(motivation_probabilities)
    
    # Normalise these motivation proportions to probabilities that add up to 1 
    if sum_motivation_probabilities > 0:
        motivation_percentages = motivation_probabilities / sum_motivation_probabilities * 100
    else:
        motivation_percentages = np.zeros(len(motivation_probabilities))

    num_bought_products = len(customer_products)  # Count of products bought by the customer

    # Store the most important motivations for this customer  
    top_motivations_indices = np.argsort(motivation_percentages)[::-1][:5]
    top_motivations_labels = [motivation_labels[idx] for idx in top_motivations_indices]
    top_motivations_percentages = [motivation_percentages[idx] for idx in top_motivations_indices]

    return i, num_bought_products, top_motivations_labels, top_motivations_percentages, motivation_percentages


# Generate segmentation for the entire set of customers:
if __name__ == '__main__':
    num_workers = 8  # Adjust the number of parallel workers as needed
    start_time = time.time()  # Record the start time

    with Pool(num_workers) as pool:
        selected_customer_indices = range(num_customers)
        results = pool.map(process_customer, selected_customer_indices)

    total_motivation_probabilities = np.zeros(phi_data.shape[1])  # Initialize an array to store total motivation probabilities

    customer_results = []

    for i, num_bought_products, top_motivations_labels, top_motivations_percentages, motivation_percentages in results:
        customer_results.append(
            {
                "CustomerID": i,
                "Number of Products Bought": num_bought_products,
                **{label: percentage for label, percentage in zip(motivation_labels, motivation_percentages)}
            }
        )

        total_motivation_probabilities += motivation_percentages  # Accumulate motivation probabilities

    average_motivation_probabilities = total_motivation_probabilities / num_customers  # Calculate the average

    print("Average Motivation Probabilities across all customers:")
    for label, percentage in zip(motivation_labels, average_motivation_probabilities):
        print(f"Motivation: {label} - Average Probability: {percentage:.2f}%")

    # Create a DataFrame from the results
    customer_results_df = pd.DataFrame(customer_results)

    # Write the DataFrame to a CSV file
    customer_results_df.to_csv("output/M30/CTM/customer_motivation_probabilities.csv", index=False)

    end_time = time.time()  # Record the end time
    elapsed_time = end_time - start_time
    print(f"Execution Time: {elapsed_time:.2f} seconds")