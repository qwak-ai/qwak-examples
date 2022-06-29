from qwak.inference.clients import RealTimeClient

if __name__ == '__main__':
    feature_vector = [
        {
            "user_id": "8b65e705-bd8e-4859-a72f-851998eb5688"
        }]

    client = RealTimeClient(model_id="risk_model_feature_store")
    print(client.predict(feature_vector))
