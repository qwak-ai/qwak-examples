# Please update MODEL_ID with your model id on Qwak
QWAK_MODEL_ID = 'your-model-id'

if __name__ == '__main__':
    from qwak_inference import RealTimeClient

    feature_vector = [
        {
            "prompt": "what is love?"
        }
    ]

    client = RealTimeClient(model_id="distilgpt2")
    client.predict(feature_vector)
