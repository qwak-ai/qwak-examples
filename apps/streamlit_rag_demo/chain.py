import streamlit as st
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from qwak_llm import Qwak


@st.cache_resource
def llm_chain_response(model_id: str) -> LLMChain:
    """
    Prepare an LLM chain with an LLM on Qwak

    :param model_id: The model ID on Qwak
    :return: LLMChain
    """

    prompt_template = """
<<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.
If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
Answer the question based on the context below
<</SYS>>


[INST]
{context}
Question: {input}[/INST]
Answer:
"""

    prompt = PromptTemplate.from_template(prompt_template)

    llm = Qwak(
        model_id=model_id
    )

    chain = LLMChain(
        prompt=prompt,
        llm=llm,
    )

    return chain
