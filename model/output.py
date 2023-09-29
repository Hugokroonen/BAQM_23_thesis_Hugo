"""
Description:
    Writes relevant output to csv

Including:
    1. Unpacking .npz file 
    2. Saving relevant parameters as .csv 
    - Theta (basket-specific motivation probabilities)
    - Phi  (motivation-product vectors)
    - Gamma (customer-specific effects)
    - Lambda_kappa (correlations)
    - Beta (trip-specific effects)
    - Kappa (alpha intercept)
"""

import pandas as pd
import numpy as np 

# 1. Unpack .npz file 


# Load the .npz file
output = np.load('output/M3/CTM/state_0000001999.npz')

# Check if "counts_phi" exists in the .npz file
if "counts_phi" in output:

    # Get the "counts_phi" array
    raw_counts_phi_df = output["counts_phi"]
    # Normalise counts_phi to ensure all product probabilities per motivation add up to 1 
    counts_phi_df = raw_counts_phi_df / raw_counts_phi_df.sum(axis=0)

    # Save "counts_phi" as a CSV file
    np.savetxt("output/M3/CTM/counts_phi.csv", counts_phi_df, delimiter=',')
    print("Saved counts_phi.csv")
 
# Check if "counts_basket" exists in the .npz file
if "counts_basket" in output:
    # Get the "counts_basket" array
    raw_counts_basket_df = output["counts_basket"]
    # Normalise basket counts to get probabilities 
    basket_counts_df = raw_counts_basket_df / raw_counts_basket_df.sum(axis=1, keepdims=True)

    # Save "counts_phi" as a CSV file
    np.savetxt("output/M3/CTM/counts_basket.csv", basket_counts_df, delimiter=',')
    print("Saved counts_basket.csv")

# Close the .npz file
output.close()



# 2. Save parameters as CSV 


# THETA: Customer basket-specific motivation relevance vectors


# Read the CSV file into a pandas DataFrame
# raw_basket_counts_df = pd.read_csv('output/M30/CTM/counts_basket.csv', sep=',', header=None)

# Normalise basket counts to get probabilities 
# basket_counts_df = raw_basket_counts_df.div(raw_basket_counts_df.sum(axis=1), axis=0)

# Create a new DataFrame to store the top motivations for each basket 
top_motivations = pd.DataFrame()
basket_counts_df = pd.DataFrame(basket_counts_df)

# Iterate over each row and find the top ... motivations 
for index, row in basket_counts_df.iterrows():
    top = row.nlargest(5)
    top_motivations[index] = top.index

# Write the DataFrame to an Excel file
top_motivations.to_csv('output/M3/CTM/basket_top_motivations.csv', index=False)



# PHI: Motivation-product vectors 


# Set the amount of associated products (P) you want to display per motivation
# For M30, the limit should be around 15, while for M3, the limit may be increased to more products
products_display_limit = 15 

# Retrieving the most important products associated with each motivation: 

# Read the raw counts_phi file into a pandas DataFrame 
# counts_phi_df = pd.read_csv('output/M30/CTM/counts_phi.csv', header=None)

# Normalise counts_phi to ensure all product probabilities per motivation add up to 1 
# counts_phi_df = raw_counts_phi_df.div(raw_counts_phi_df.sum(axis=0), axis=1)

# Create a new DataFrame to store the most probable P product id's for each motivation 
motivations_df = pd.DataFrame()
counts_phi_df = pd.DataFrame(counts_phi_df)


# For each motivation, find the most probable P products 
for column in counts_phi_df.columns:
    top_product_indices = counts_phi_df.nlargest(products_display_limit, column, keep='first')
    motivations_df[column] = top_product_indices.index

# Read the product data from 'data/final_sample.csv' into a DataFrame
df_products = pd.read_csv('data/products.csv')

# Create a dictionary to map 'product_id' to 'lowest_level'
product_id_to_lowest_level_dict = df_products.drop_duplicates(subset='product_id').set_index('product_id')['lowest_level'].to_dict()

# Map the product IDs to the product names in top_10_products
motivations_top_products_df = motivations_df.apply(lambda col: col.map(product_id_to_lowest_level_dict))

# Write the top 10 products for each motivation to a new Excel file
motivations_top_products_df.to_csv('output/M3/CTM/motivations_top_products.csv', index=False)

# Retrieving the corresponding probabilities: 

# Calculate the sum of each column in the original DataFrame
counts_phi_summed_df = counts_phi_df.sum()

# Create a new DataFrame to store the proportions of the most probable P product numbers for each motivation 
motivation_product_probabilities_df = pd.DataFrame()

# For each motivation, find the most probable X products and calculate the proportions
for column in counts_phi_df.columns:
    top_product_values = counts_phi_df.nlargest(products_display_limit, column, keep='first')
    top_product_probabilities = top_product_values[column] / counts_phi_summed_df[column]
    motivation_product_probabilities_df[column] = top_product_probabilities.values  # Extract the proportions without indexes

# Write the proportions for each motivation to a new Excel file
motivation_product_probabilities_df.to_csv('output/M3/CTM/motivations_with_product_probabilities.csv', index=False)

# Read the two CSV files into pandas DataFrames
motivation_product_probabilities_df = pd.read_csv('output/M3/CTM/motivations_with_product_probabilities.csv')
motivations_products_df = pd.read_csv('output/M3/CTM/motivations_top_products.csv')

# Get the number of rows in each DataFrame
num_rows = len(motivation_product_probabilities_df)

# Create a new DataFrame to store the blended data
motivations_final_df = pd.DataFrame()

# Interleave the columns from both DataFrames
for col_name, col_proportions in motivation_product_probabilities_df.items():
    col_product_names = motivations_products_df[col_name]
    motivations_final_df[col_name + '_product_names'] = col_product_names
    motivations_final_df[col_name + '_proportions'] = col_proportions

# Write the blended DataFrame to a new CSV file
motivations_final_df.to_csv('output/M3/CTM/motivations_result.csv', index=False)


# Load the .npz file again 
output = np.load('output/M3/CTM/state_0000001999.npz')


# GAMMA: customer-specific effects 


# Access the 'gamma' array
gamma_array = output['gamma']

# Convert the 'gamma' array to a pandas DataFrame
gamma_df = pd.DataFrame(gamma_array)

# Write the DataFrame to an Excel file
gamma_df.to_csv('output/M3/CTM/gamma.csv', index=False)


# LAMBDA-KAPPA: correlations 


# Access the 'lambda kappa' array
lambda_kappa_array = output['lambda_kappa']

# Convert the 'lambda kappa ' array to a pandas DataFrame
lambda_kappa_df = pd.DataFrame(lambda_kappa_array)

# Write the DataFrame to an Excel file
lambda_kappa_df.to_csv('output/M3/CTM/lambda_kappa.csv', index=False)


# BETA: trip-specific effects 


# Access the 'beta' array
beta_array = output['beta']

# Convert the 'beta' array to a pandas DataFrame
beta_df = pd.DataFrame(beta_array)

# Write the DataFrame to an Excel file
beta_df.to_csv('output/M3/CTM/beta.csv', index=False) 


# KAPPA: intercept 


# Access the 'kappa' array
lambda_kappa_array = output['kappa']

# Convert the 'kappa' array to a pandas DataFrame
lambda_kappa_df = pd.DataFrame(lambda_kappa_array)

# Write the DataFrame to an Excel file
lambda_kappa_df.to_csv('output/M3/CTM/kappa.csv', index=False)