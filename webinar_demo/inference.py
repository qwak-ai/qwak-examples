from qwak_inference import RealTimeClient

SENTENCE_EMBEDDINGS_MODEL_ID = 'sentence_embeddings'
FLAN_T5_FINETUNED_MODEL_ID = 'finetuned_flan_t5'
# FLAN_T5_FINETUNED_MODEL_ID = 'flan_t5_fine_tuned'  # Old model
FLAN_T5_MODEL_ID = 'flan_t5_large'
FALCON_7B_MODEL_ID = 'falcon_7b'
PYTHIA_MODEL_ID = 'parameter_efficient_fine_tuning'


def generate_embeddings(input_text: str, model_id=SENTENCE_EMBEDDINGS_MODEL_ID):
    feature_vector = [{
        'text': input_text
    }]
    client = RealTimeClient(model_id=model_id)
    response = client.predict(feature_vector)
    return response[0]


def flan_completion(input_text: str, model_id=FLAN_T5_MODEL_ID):
    feature_vector = [{
        'prompt': "Question: " + input_text
    }]
    client = RealTimeClient(model_id=model_id)
    response = client.predict(feature_vector)
    return response[0]["generated_text"][0]


def falcon_completion(input_text: str, model_id=FALCON_7B_MODEL_ID):
    feature_vector = [{
        'prompt': "Provide a short answer: " + input_text
    }]
    client = RealTimeClient(model_id=model_id)
    response = client.predict(feature_vector)
    return response[0]['0']["generated_text"]


def peft_completion(input_text: str, model_id=PYTHIA_MODEL_ID):
    feature_vector = [{
        'prompt': input_text
    }]
    client = RealTimeClient(model_id=model_id)
    response = client.predict(feature_vector)
    return response[0]["generated_text"]