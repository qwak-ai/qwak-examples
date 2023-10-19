from qwak_inference import RealTimeClient
from pandas import DataFrame


model_id = 'llama2'
prompt = "Answer: how to bake an apple pie?"
input_ = DataFrame([{"prompt": prompt}]).to_json()

client = RealTimeClient(model_id=model_id)

response = client.predict(input_)
print(response)