from functools import partial
from typing import Callable

import streamlit as st
from PIL import Image
from streamlit_chat import message
from streamlit_extras.add_vertical_space import add_vertical_space

from inference import generate_embeddings, FLAN_T5_FINETUNED_MODEL_ID, FLAN_T5_MODEL_ID,\
    get_qwak_token, get_api_inference

# API_KEY = 'your-key'
API_KEY = 'eu-5731d1422fb34029b6e60b702ddfb9a3@5m))6Im#MAaDlL0eV1ahL990))ge^9t-'


# Fetch a token upon startup
if 'qwak_token' not in st.session_state:
    st.session_state['qwak_token'] = get_qwak_token(API_KEY)


MODELS = {
    "FLAN T5": {
        "model_id": FLAN_T5_MODEL_ID,
        "fn": partial(get_api_inference, qwak_token=st.session_state['qwak_token'])
    },
    "Finetuned T5": {
        "model_id": FLAN_T5_FINETUNED_MODEL_ID,
        "fn": partial(get_api_inference, qwak_token=st.session_state['qwak_token'])
    },
}


def show_image(image_path):
    image = Image.open(image_path)
    st.image(image, use_column_width="always")


def generate_response(prompt: str, model_id: str, inference_fn: Callable):
    print(prompt, model_id)
    return inference_fn(model_input=prompt, model_id=model_id)


def get_text():
    return st.text_input(label="Ask Qwak: ", placeholder="", key="input")


st.set_page_config(page_title="LLMs on Qwak")
show_image('llm_cover.png')
add_vertical_space(2)

col1, col2 = st.columns([1, 3])

with st.container():
    with col1:
        current_model = st.selectbox(
            'Using Qwak Model',
            MODELS.keys(),
        )
    with col2:
        user_input = get_text()

add_vertical_space(2)

with st.container():
    input_container = st.container()
    response_container = st.container()

    with input_container:
        if user_input:
            message(user_input, is_user=True)

    with response_container:
        if user_input:
            with st.spinner('Loading...'):
                response = generate_response(
                    prompt=user_input,
                    model_id=MODELS[current_model]["model_id"],
                    inference_fn=MODELS[current_model]["fn"]
                )
                if response:
                    message(response)

            add_vertical_space(2)
            with st.container():
                with st.spinner("Generating conversation embeddings..."):
                    embedding_input = f"Question: {user_input} Answer: {response}"
                    embeddings = generate_embeddings(input_text=embedding_input,
                                                     qwak_token=st.session_state['qwak_token'])
                    if embeddings:
                        with st.expander("View embeddings"):
                            st.write(embedding_input)
                            st.write(embeddings)
