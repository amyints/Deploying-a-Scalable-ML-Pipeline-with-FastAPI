import pytest
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from ml.data import process_data
from ml.model import (
    compute_model_metrics,
    inference,
    load_model,
    performance_on_categorical_slice,
    save_model,
    train_model,
)

# TODO: add necessary import
@pytest.fixture
def sample_data():
    """
    a fixture to provide a sample dataset for the tests.
    """
    data = pd.read_csv('./data/census.csv')
    return data.head(100)

# TODO: implement the first test. Change the function name and input as needed
def test_one(sample_data):
    """
    Test to check if the training and test datasets have the expected size and data type
    """
    # Your code here\

    train, test = train_test_split(sample_data, test_size=0.2, random_state=42)

    # expected values
    expected_train_size = (80, 15)
    expected_test_size = (20, 15)

    # check that the training dataset is of the correct type and size
    assert isinstance(train, pd.DataFrame), f"Expected train to be of type pd.DataFrame, but got {type(train)}"
    assert train.shape == expected_train_size, f"Expected train size to be {expected_train_size}, but got {train.shape}"

    # check that the test dataset is of the correct type and size
    assert isinstance(test, pd.DataFrame), f"Expected test to be of type pd.DataFrame, but got {type(test)}"
    assert test.shape == expected_test_size, f"Expected train size to be {expected_test_size}, but got {test.shape}"






# TODO: implement the second test. Change the function name and input as needed
def test_two(sample_data):
    """
    # add description for the second test
    """
    # Your code here
    
    # Prepare the train
    X_train, y_train, encoder, lb = process_data(
        sample_data,
        categorical_features=['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country'],
        label='salary',
        training=True
    )

    # Train the model using the expected algorithm (RandomForestClassifier)
    model = train_model(X_train, y_train)

    assert isinstance(model, RandomForestClassifier), F"Expected model to be RandomForestClassifier, but got {type(model)}"



# TODO: implement the third test. Change the function name and input as needed
def test_three():
    """
    # add description for the third test
    """
    # Your code here
    pass
