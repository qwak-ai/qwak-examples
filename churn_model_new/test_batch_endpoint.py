from frogml_inference import BatchInferenceClient
import pandas as pd
import os

JFROGML_MODEL_ID = 'churn_prediction_model'

# You can also set the QWAK_MODEL_ID environment variable instead of passing it
batch_inference_client = BatchInferenceClient(model_id=JFROGML_MODEL_ID)

file_absolute_path = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(f"{file_absolute_path}/main/data.csv")


# You should pass the DataFrame and batch size, and the others will use the deployed configuration
result_df = batch_inference_client.run(
    df, # mandatory
    batch_size=100, # mandatory
    executors=1,
    instance='tiny')


print(len(result_df))