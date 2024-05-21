if 'custom' not in globals():
    from mage_ai.data_preparation.decorators import custom
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pickle

@custom
def transform_custom(*args, **kwargs):
    """
    args: The output from any upstream parent blocks (if applicable)

    Returns:
        Anything (e.g. data frame, dictionary, array, int, str, etc.)
    """
    # Load the model and vectorizer from the file
    with open('models/lin_reg.bin', 'rb') as f_in:
        vec, model = pickle.load(f_in)

    # Read new dataset
    url_feb = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-02.parquet"
    df = pd.read_parquet(url_feb)

    # Feature Engineering on the new dataset
    categorical = ['PULocationID', 'DOLocationID']
    numerical = ['trip_distance']

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)
    df = df[(df.duration >= 1) & (df.duration <= 60)]

    target = 'duration'

    df[categorical] = df[categorical].astype(str)
    
    dv = DictVectorizer()

    val_dicts = df[categorical + numerical].to_dict(orient='records')
    X_val = vec.transform(val_dicts)
    y_val = df[target].values

    # Make predictions using the loaded model
    y_pred = model.predict(X_val)

    rmse = mean_squared_error(y_val, y_pred, squared=False)
    # Print the RMSE
    print(f"Root Mean Squared Error (RMSE): {rmse}")
    return {}


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'
