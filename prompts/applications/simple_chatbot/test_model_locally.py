import json
from pandas import DataFrame
from qwak.model.tools import run_local
from main import *

if __name__ == '__main__':

    model = load_model()

    input_vector = DataFrame(
        [{
            "prompt": "Tell me a funny joke",
        }]
    ).to_json()

    prediction = run_local(model, input_vector)
    prediction_data = json.loads(prediction)

    content = prediction_data[0]["generated_text"]
    print(f"\n\nChatbot Response:\n\n{content}")
