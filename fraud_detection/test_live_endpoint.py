import pandas as pd
from frogml_inference import RealTimeClient

JFROGML_MODEL_ID = 'fraud_detection_model'

if __name__ == '__main__':

    # Load the data
 
    input_ = pd.read_csv("main/small_fraud_dataset.csv", nrows=1000)
 
    client = RealTimeClient(model_id=JFROGML_MODEL_ID)
    
    response = client.predict(input_)
    print(pd.DataFrame(response))
