"""
Description:
    In this script, we evaluate the hit rates of each recommmender system. 
"""

import pandas as pd

# Load the hold-out purchases and prediction set (adjust filenames as needed)
holdout_purchases = pd.read_csv("data/y_test.csv", header=None)

# Select prediction set to analyse

prediction_set = pd.read_csv("data/prediction_set_CTM_30.csv", header=None)
# prediction_set = pd.read_csv("data/prediction_set_restricted_CTM_30.csv", header=None)
# prediction_set = pd.read_csv("data/prediction_set_marginal.csv", header=None)
# prediction_set = pd.read_csv("data/prediction_set_popularity.csv", header=None)


# Calculate hit rate for each customer separately
hit_rates = []
for customer_id in prediction_set[0].unique():
    customer_prediction = prediction_set[prediction_set[0] == customer_id]
    customer_holdout = holdout_purchases[holdout_purchases[0] == customer_id]
    overlap = len(customer_prediction[customer_prediction[2].isin(customer_holdout[2])])

    S = len(customer_prediction)
    u = len(customer_holdout[2].unique())
    max_hits = min(S, u)
    if max_hits > 0:
        customer_hit_rate = overlap / max_hits
        hit_rates.append(customer_hit_rate)

# Calculate the average hit rate across all customers 
average_hit_rate = sum(hit_rates) / len(hit_rates) if len(hit_rates) > 0 else 0

print("Average Hit Rate:", average_hit_rate)
