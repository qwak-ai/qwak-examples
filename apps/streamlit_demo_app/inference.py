import requests

SENTENCE_EMBEDDINGS_MODEL_ID = 'sentence_embeddings'
FLAN_T5_FINETUNED_MODEL_ID = 'fine_tuned_flan_t5'
FLAN_T5_MODEL_ID = 'flan_t5'
FALCON_7B_MODEL_ID = 'falcon_7b'
PYTHIA_MODEL_ID = 'parameter_efficient_fine_tuning'


def get_qwak_token(api_key: str):
    """
    Uses Qwak REST API to get an access token
    :param api_key: Your Qwak API Key
    :return: a valid Qwak token
    """
    response = requests.post(
        url='https://grpc.qwak.ai/api/v1/authentication/qwak-api-key',
        headers={
            'Content-Type': 'application/json',
        },
        json={'qwakApiKey': api_key}
    )
    if response.ok:
        return response.json()['accessToken']
    else:
        raise ValueError('Missing token')


def get_api_inference(model_input: str, model_id: str, qwak_token: str = None, api_key: str = None):
    """
    Uses Qwak REST API to fetch model predictions
    :param model_input: The input string
    :param model_id: The model ID to fetch
    :param qwak_token: A valid Qwak token. If kept None, a new one is generated
    :param api_key: Optional API key to fetch a token on the fly
    :return: A valid model prediction
    """
    if not qwak_token:
        qwak_token = get_qwak_token(api_key)

    prompt = "question: " + model_input

    res = requests.post(
        url=f'https://models.llm-demo.qwak.ai/v1/{model_id}/predict',
        headers={
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {qwak_token}'
        },
        json={
            'columns': ["prompt"],
            'index': [0],
            "data": [[prompt]]
        }
    )

    return res.json()[0]["generated_text"][0]


def generate_embeddings(input_text: str, model_id=SENTENCE_EMBEDDINGS_MODEL_ID, qwak_token: str = None):

    prompt = "Question: " + input_text

    res = requests.post(
        url=f'https://models.llm-demo.qwak.ai/v1/{model_id}/predict',
        headers={
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {qwak_token}'
        },
        json={
            'columns': ["text"],
            'index': [0],
            "data": [[prompt]]
        }
    )

    return res.json()


def flan_completion(input_text: str, model_id=FLAN_T5_MODEL_ID):
    from qwak_inference import RealTimeClient

    feature_vector = [{
        'prompt': "Question: " + input_text
    }]
    client = RealTimeClient(model_id=model_id)
    response = client.predict(feature_vector)
    return response[0]["generated_text"][0]


def falcon_completion(input_text: str, model_id=FALCON_7B_MODEL_ID):
    from qwak_inference import RealTimeClient

    feature_vector = [{
        'prompt': "Provide a short answer: " + input_text
    }]
    client = RealTimeClient(model_id=model_id)
    response = client.predict(feature_vector)
    return response[0]['0']["generated_text"]


def peft_completion(input_text: str, model_id=PYTHIA_MODEL_ID):
    from qwak_inference import RealTimeClient

    feature_vector = [{
        'prompt': "question: " + input_text
    }]
    client = RealTimeClient(model_id=model_id)
    response = client.predict(feature_vector)
    return response[0]["generated_text"]
