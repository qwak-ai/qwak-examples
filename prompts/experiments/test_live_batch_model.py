import argparse
from qwak_inference.batch_client.batch_client import BatchInferenceClient
import pandas as pd


def main(model_id):

    df = pd.read_csv('./main/natural_language_to_sql.csv')
    new_df = df.drop("query", axis=1)

    batch_inference_client = BatchInferenceClient(model_id=model_id)
    result = batch_inference_client.run(
        new_df,
        batch_size=500,
        executors=1,
        instance="small",
        serialization_format="PARQUET"
    )

    print(result)


"""
USAGE:

>> python main/test_live_batch_model.py <your_model_id>

"""
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict using the following Qwak model-id.')
    parser.add_argument('model_id', type=str, help='The Qwak model ID to call for prediction.')

    args = parser.parse_args()
    main(args.model_id)