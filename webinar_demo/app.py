from functools import partial
from typing import Callable

import streamlit as st
from PIL import Image
from streamlit_chat import message
from streamlit_extras.add_vertical_space import add_vertical_space

from inference import falcon_completion, flan_completion, generate_embeddings, \
    FLAN_T5_FINETUNED_MODEL_ID, FLAN_T5_MODEL_ID, FALCON_7B_MODEL_ID

MODELS = {
    "FLAN T5": {
        "model_id": FLAN_T5_MODEL_ID,
        "fn": flan_completion
    },
    "Finetuned T5": {
        "model_id": FLAN_T5_FINETUNED_MODEL_ID,
        "fn": partial(flan_completion, model_id=FLAN_T5_FINETUNED_MODEL_ID)
    },
    "Falcon 7b": {
        "model_id": FALCON_7B_MODEL_ID,
        "fn": falcon_completion
    }
}


def show_image(image_path):
    image = Image.open(image_path)
    st.image(image, use_column_width="always")


def generate_response(prompt: str, model_id: str, inference_fn: Callable):
    # print(model_id, prompt)
    return inference_fn(prompt)


def get_text():
    return st.text_input(label="Ask Qwak: ", placeholder="", key="input")


st.set_page_config(page_title="LLMs on Qwak")
show_image('llm_cover_1.png')
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
                response = generate_response(user_input, current_model, MODELS[current_model]["fn"])
                if response:
                    message(response)

            add_vertical_space(2)
            with st.container():
                with st.spinner("Generating conversation embeddings..."):
                    embedding_input = f"Question: {user_input} Answer: {response}"
                    embeddings = generate_embeddings(embedding_input)
                    if embeddings:
                        with st.expander("View embeddings"):
                            st.write(embedding_input)
                            st.write(embeddings)
