"""
Running example.
The files in the path dataset show an example of how you need to model the entrance.
"""
import pandas as pd

from scikit_pierre.metrics.evaluation import MeanAbsoluteCalibrationError, MeanAveragePrecision
from scikit_pierre.tradeoff.calibration import PopularityCalibration, LinearCalibration

# Load the users' preference set as a Pandas DataFrame instance.
# It is expected 3 columns: [USER_ID, ITEM_ID, TRANSACTION_VALUE].
users_preference_dataframe = pd.read_csv('dataset/train.csv')

# Load the users' candidate items set as a Pandas DataFrame instance.
# It is expected 3 columns: [USER_ID, ITEM_ID, TRANSACTION_VALUE].
candidate_items_dataframe = pd.read_csv('dataset/candidate_items.csv')

candidate_items_top_10 = pd.concat(
    [
        df.sort_values(by="TRANSACTION_VALUE", ascending=False).head(10)
        for ix, df in candidate_items_dataframe.groupby(by="USER_ID")
    ]
)


test_items_dataframe = pd.read_csv('dataset/test.csv')

# Load the items set as a Pandas DataFrame instance.
# It is expected 2 columns: [ITEM_ID, GENRES].
items_dataframe = pd.read_csv('dataset/items.csv')

from scikit_pierre.models.item import ItemsInMemory
items_inst = ItemsInMemory(data=items_dataframe)
items_inst.classifying_item_by_popularity(users_transactions=users_preference_dataframe)

items_dataframe = items_inst.transform_to_pandas_items()
# items_dataframe.rename(columns={"GENRES": "POPULARITY"}, inplace=True)

# ################################################# #
# ################################################# #
# ################################################# #
print("Computing first stage ... Popularity ...")

# Create an instance with the basic data.
pop_instance = PopularityCalibration(
    users_preferences=users_preference_dataframe,
    candidate_items=candidate_items_dataframe,
    item_set=items_dataframe
)

# Configure the instance
pop_instance.config(
    distribution_component='CWS',
    fairness_component='KL',
    relevance_component='SUM',
    tradeoff_weight_component='VAR',
    select_item_component='SURROGATE',
    list_size=50
)

# Execute the instance and get the recommendation list to all users.
pop_recommendation_lists = pop_instance.fit()
# print the 5 first in the dataframe
print(pop_recommendation_lists.head(5))

mace_pop = MeanAbsoluteCalibrationError(
    users_profile_df=users_preference_dataframe,
    users_rec_list_df=pop_recommendation_lists,
    items_set_df=items_dataframe
)

mace_value_pop = mace_pop.compute()

print(f"MACE Value is {mace_value_pop}.")

map_pop = MeanAveragePrecision(
    users_rec_list_df=pop_recommendation_lists,
    users_test_set_df=test_items_dataframe
)

map_pop_value = map_pop.compute()

map_original = MeanAveragePrecision(
    users_rec_list_df=candidate_items_dataframe,
    users_test_set_df=test_items_dataframe
)

map_original_value = map_original.compute()
print(f"MAP Value is {map_pop_value} and the original map is {map_original_value}.")

# ################################################ #
# ################################################ #
# ################################################ #
print("Computing second stage ... Genres ...")
items_dataframe = pd.read_csv('dataset/items.csv')

# Create an instance with the basic data.
calib_instance = LinearCalibration(
    users_preferences=users_preference_dataframe,
    candidate_items=candidate_items_dataframe,
    item_set=items_dataframe
)

# Configure the instance
calib_instance.config(
    distribution_component='CWS',
    fairness_component='KL',
    relevance_component='SUM',
    tradeoff_weight_component='VAR',
    select_item_component='SURROGATE',
    list_size=10
)

# Execute the instance and get the recommendation list to all users.
calib_rec_list = calib_instance.fit()
# print the 5 first in the dataframe
print(calib_rec_list.head(5))

mace = MeanAbsoluteCalibrationError(
    users_profile_df=users_preference_dataframe,
    users_rec_list_df=calib_rec_list,
    items_set_df=items_dataframe
)

mace_value = mace.compute()

print(f"MACE Value is {mace_value}.")

map_calibration = MeanAveragePrecision(
    users_rec_list_df=calib_rec_list,
    users_test_set_df=test_items_dataframe
)

map_calibration_value = map_calibration.compute()

print(f"MAP Value is {map_calibration_value} and the original map is {map_original_value}.")