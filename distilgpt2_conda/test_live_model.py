from qwak_inference import RealTimeClient

# Please update MODEL_ID with your model id on Qwak
QWAK_MODEL_ID = 'your-model-id'

if __name__ == '__main__':
    input_ = {"prompt": "what is love?"}
    client = RealTimeClient(model_id=QWAK_MODEL_ID)
    response = client.predict(input_)
    print(response)
