"""
Description:
    In this script, we perform preparations to evaluate the novelty hit rate of the recommender systems. As such, we find
    the novel purchases in each customer's test set, and keep only those.  
"""

import pandas as pd

# Load the hold-out purchases (test set)
holdout_purchases = pd.read_csv("data/y_test.csv", header=None)

# Load the training purchases
training_purchases = pd.read_csv("data/y_training.csv", header=None)

novel_purchases_per_customer = []

for customer_id in holdout_purchases[0].unique():
    customer_test = holdout_purchases[holdout_purchases[0] == customer_id]
    customer_training = training_purchases[training_purchases[0] == customer_id]

    # Find new/novel purchases in the test set
    novel_purchases = customer_test[~customer_test[2].isin(customer_training[2])]
    if not novel_purchases.empty:
        for _, row in novel_purchases.iterrows():
            novel_purchases_per_customer.append((customer_id, row[2]))

# Convert the results to a DataFrame
novel_purchases_df = pd.DataFrame(novel_purchases_per_customer, columns=['customer_id', 'product_id'])

# Write the DataFrame to a CSV file
novel_purchases_df.to_csv('data/y_test_novel.csv', index=False, header=None)
