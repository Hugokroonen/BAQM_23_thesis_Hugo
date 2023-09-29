"""
Description:
    Splits the sample into training and test set, with the test set consisting of each customer's last shopping trip. 
    To mimic 'new' customers for evaluation purposes, we also create a training set for new customers, consisting of each
    customer's second-to-last shopping trip.
"""

import pandas as pd

# Regular train/test split 

# Load the dataset without specifying column names
data = pd.read_csv("data/y_segmentation.csv", header=None)

# Identify the last basket for each customer
last_baskets = data.groupby(0)[1].tail(1) 

# Test set is the last shopping trip 
test_set = data[data[1].isin(last_baskets)] 

# Exclude the last baskets from the training set
training_set = data[~data.index.isin(test_set.index)]

# Reset indexes
training_set.reset_index(drop=True, inplace=True)
test_set.reset_index(drop=True, inplace=True)

# Save the training set and test set to separate CSV files
training_set.to_csv("data/y_training.csv", index=False, header=False)
test_set.to_csv("data/y_test.csv", index=False, header=False)

# Training set to mimic 'new' customers 

# Identify the second-to-last basket for each customer
second_to_last_baskets = data.groupby(0)[1].nth(-2)

# Create training_set_new_customers by excluding the second-to-last baskets from the training set
training_set_new_customers = data[~data.index.isin(second_to_last_baskets.index)]

# Reset index
training_set_new_customers.reset_index(drop=True, inplace=True)

# Save the training set and test set to separate CSV files
training_set_new_customers.to_csv("data/y_training_new_customers.csv", index=False, header=False)