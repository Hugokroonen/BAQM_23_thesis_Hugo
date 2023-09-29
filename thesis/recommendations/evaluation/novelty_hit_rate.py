"""
Description:
    In this script, we evaluate the novelty hit rates of each recommmender system. 
"""

import pandas as pd

# The test set now only contains new/novel purchases 
holdout_novel_purchases = pd.read_csv("data/y_test_novel.csv", header=None)

# Uncomment the prediction set to analyse

# prediction_set = pd.read_csv("data/prediction_set_CTM_30.csv", header=None)
prediction_set = pd.read_csv("data/prediction_set_restricted_CTM_30.csv", header=None)
# prediction_set = pd.read_csv("data/prediction_set_marginal.csv", header=None)
# prediction_set = pd.read_csv("data/prediction_set_popularity.csv", header=None)

# Calculate novelty hit rate for each customer separately
hit_rates = []
for customer_id in prediction_set[0].unique():
    customer_prediction = prediction_set[prediction_set[0] == customer_id]
    customer_novel_holdout = holdout_novel_purchases[holdout_novel_purchases[0] == customer_id] 

    if len(customer_novel_holdout) > 0: 
        overlap = len(customer_prediction[customer_prediction[2].isin(customer_novel_holdout[1])])

        S = len(customer_prediction)
        u = len(customer_novel_holdout[1].unique())
        max_hits = min(S, u)
        if max_hits > 0:
            customer_hit_rate = overlap / max_hits
            hit_rates.append(customer_hit_rate)

# Calculate the average hit rate across all customers 
average_hit_rate = sum(hit_rates) / len(hit_rates) if len(hit_rates) > 0 else 0

print("Average Hit Rate on Novel Purchases:", average_hit_rate)
