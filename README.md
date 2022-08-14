# Scikit Pierre

[![Version](https://img.shields.io/badge/version-v0.1-green)](https://github.com/scikit-pierre/scikit-pierre) ![PyPI - Python Version](https://img.shields.io/badge/python-3.8-blue)

[Docs](https://readthedocs.io/) | [Paper](https://doi.org/xx.xxx/xxxxx.xxxxxx)

[![logo](Pierre.svg)](https://project-pierre.github.io/project-pierre)

## Overview

Scikit-Pierre is a **Sci**entific Tool**Kit** for **P**ost-process**i**ng **Re**commendations. Written in Python, the library is focused on the researcher's perspective, providing an experimental environment based on the state-of-the-art. Pierre works through a pass data and returns results. It receives a set of data and from it generates a recommendation list as return.

Inspired by [scikit-learn](https://scikit-learn.org/stable/) and [Surpise](http://surpriselib.com/), Scikit-Pierre is composed of post-processing recommendation methods that are provided as api-classes.

- _First of all_, **instance** one class and pass the data.
- _Second_, **set up** the experiment with context functions.
- _Third_, **fit** the context and receive the recommendation list.

## Attention

Pierre is a free library and has not direct grant, i.e., it is made with love and science. By this, please if you find some error/issue, remember that we have no grant support and demonstrate some kind of education in contact us. We will investigate the error and try to solve.

## Built by

The original code draft is made by [Diego Corrêa da Silva](https://github.com/DiegoCorrea) as part of his master dissertation and PhD thesis at Federal University of Bahia❤️.

## Development Status

Pierre was designed with the following contexts in mind:

- Version 0.1 comprehends the context of **Calibrated Recommendations**.
- Version 0.2 comprehends a new implementations. (Incoming)

## Install and Config

With python and pip do:

    $ pip install scikit-pierre

For the latest version, you can also clone the repo and build the source:

    $ git clone https://github.com/scikit-pierre/scikit-pierre
    $ cd scikit-pierre
    $ python setup.py sdist bdist_wheel
    $ pip install dist/scikit_pierre-0.0.1_build1-py3-none-any.whl

Windows users might prefer using conda. **We do not support Windows issues**, we will try, but...

## Usage example

```python
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
for user_recommendation in recommendation_lists[:2]:
    # print the 5 first in the dataframe
    print(user_recommendation.head(5))
```
**Output**:

```
   ITEM_ID  ORDER  USER_ID  TRANSACTION_VALUE
0       53      1        1           4.752746
1      905      2        1           4.632595
2     3365      3        1           4.534034
3     2565      4        1           4.499202
4     3077      5        1           4.426636
   ITEM_ID  ORDER  USER_ID  TRANSACTION_VALUE
0     2762      1        2           4.560991
1     1035      2        2           4.390060
2      953      3        2           4.366187
3      905      4        2           4.165182
4     3365      5        2           4.128045

```

## Acknowledgments

We need to thank some import support to construct the Pierre:

- **Student grant**

  - The Coordenação de Aperfeiçoamento de Pessoal de Nível Superior - Brasil (CAPES) - Finance Code 88887.502736 / 2020–00. For provide the Diego's master degree grant;
  - The Coordenação de Aperfeiçoamento de Pessoal de Nível Superior - Brasil (CAPES) - Finance Code 88887.685243 / 2022-00. For provide the Diego's Ph.D. degree grant;

- **Machine support**

  - The Brazilian National Laboratory of Scientific Computing (LNCC/MCTI, Brazil ❤️ ), URL: http://sdumont.lncc.br. For provide the access to the [Santos Dumont HPC](https://www.top500.org/system/179704/);
  - The Center for Computational Biology and Biotechnological Information Management - NBCGI, with resources from FINEP/ MCT, CNPQ and FAPESB and from the State University of Santa Cruz - UESC.

## People Contribution

- [Diego Corrêa da Silva]()

# References

[Docs]: Incoming  
[Paper]: Incoming  
