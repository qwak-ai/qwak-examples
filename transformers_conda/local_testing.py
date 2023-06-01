from qwak_inference import RealTimeClient

# Please update MODEL_ID with your model id on Qwak
QWAK_MODEL_ID = 'your-model-id'

if __name__ == '__main__':
    feature_vector = [{
        'text': 'This is the best place ever!'
    }]

    client = RealTimeClient(model_id=QWAK_MODEL_ID)
    response = client.predict(feature_vector)
    print(response)
