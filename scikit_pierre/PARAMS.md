:param users_preferences: A Pandas DataFrame with four columns [USER_ID, ITEM_ID, TRANSACTION_VALUE, TIMESTAMP].
:param candidate_items: A Pandas DataFrame with three columns [USER_ID, ITEM_ID, PREDICTED_VALUE].
:param item_set: A Pandas DataFrame of items with two columns [ITEM_ID, GENRES].
:param p: A list with float numbers, which represents the distribution values, p and q need to be the same size.
:param q: A list with float numbers, which represents the distribution values, p and q need to be the same size.
:param alpha: Trade-off weight value to Realized distribution \tilde{q}


:return: A list with floats numbers that represent the new realized distribution values.
:return: A Pandas DataFrame with the USER_ID as index, GENRES as columns,
            and the distribution value as cells.