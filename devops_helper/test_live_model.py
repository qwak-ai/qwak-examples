import pandas as pd
from qwak_inference import RealTimeClient

JFML_MODEL_ID = "finetuned_qwen"


if __name__ == '__main__':

    # Define the columns
    columns = ["prompt"]

    # Define the data
    data = [["How do I expose a deployment in Kubernetes using a service?"]]
    
    # Create the DataFrame and convert it to JSON
    _input = pd.DataFrame(data, columns=columns).to_json(orient='records')
 
    client = RealTimeClient(model_id=JFML_MODEL_ID)
    
    print (_input)
    response = client.predict(_input)
    print(pd.DataFrame(response))