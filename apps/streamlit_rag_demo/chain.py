import streamlit as st
from langchain.chains import LLMChain
from langchain.memory import ConversationSummaryBufferMemory
from langchain.prompts import PromptTemplate
from qwak_llm import Qwak


@st.cache_resource
def llm_chain_response(model_id: str) -> LLMChain:
    """
    Prepare an LLM chain with an LLM on Qwak

    :param model_id: The model ID on Qwak
    :return: LLMChain
    """
    prompt = PromptTemplate(
        input_variables=[
            "history",
            "input"
        ],
        template="Answer the question based on the context below."
                 " If you cannot answer based on the context truthfully, answer that you don't know."
                 " Use Markdown and text formatting to format your answer. "
                 "\n\nCurrent conversation:\n{history}\nHuman: {input}\nAI:"
    )

    llm = Qwak(
        model_id=model_id
    )

    chat_chain = LLMChain(
        llm=llm,
        prompt=prompt,
        memory=ConversationSummaryBufferMemory(
            llm=llm,
            max_token_limit=256
        )
    )
    return chat_chain