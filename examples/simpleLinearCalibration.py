from scikit_pierre.tradeoff.calibration import LinearCalibration
import pandas as pd

# Load the users' preference set as a Pandas DataFrame instance.
# It is expected 3 columns: [USER_ID, ITEM_ID, TRANSACTION_VALUE].
users_preference_dataframe = pd.read_csv('dataset/train.csv')

# Load the users' candidate items set as a Pandas DataFrame instance.
# It is expected 3 columns: [USER_ID, ITEM_ID, TRANSACTION_VALUE].
candidate_items_dataframe = pd.read_csv('dataset/candidate_items.csv')

# Load the items set as a Pandas DataFrame instance.
# It is expected 2 columns: [ITEM_ID, GENRES].
items_dataframe = pd.read_csv('dataset/items.csv')

# Create an instance with the basic data.
tradeoff_instance = LinearCalibration(
  users_preferences=users_preference_dataframe,
  candidate_items=candidate_items_dataframe,
  item_set=items_dataframe
)

# Configure the instance
tradeoff_instance.config(
  distribution_component='CWS',
  fairness_component='KL',
  relevance_component='SUM',
  tradeoff_weight_component='VAR',
  select_item_component='SURROGATE',
  list_size=10
)

# Execute the instance and get the recommendation list to all users.
recommendation_lists = tradeoff_instance.fit()
# print the 5 first in the dataframe
print(recommendation_lists.head(5))
