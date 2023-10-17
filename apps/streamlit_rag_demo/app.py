import streamlit as st
from langchain.chains import LLMChain
from langchain.memory import ConversationSummaryBufferMemory
from langchain.prompts import PromptTemplate
from qwak.vector_store import VectorStoreClient
from streamlit_chat import message

from qwak_llm import QwakLLM


@st.cache_resource
def llm_chain_response():
    prompt = PromptTemplate(
        input_variables=[
            "history",
            "input"
        ],
        template="Answer the question based on the context below. If you cannot answer based on the context "
                 "truthfully, answer that you don't know."
                 " Use Markdown and text formatting to format your answer. "
                 "\n\nCurrent conversation:\n{history}\nHuman: {input}\nAI:"
    )

    llm = QwakLLM(
        model_id="flan_t5"
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


def retrieve(query, collection_name: str = "articles") -> str:
    """
    Retrieve the context from the Vector Store
    :param query: User query
    :param collection_name: Name of the vector store collection
    :return: The full context
    """
    client = VectorStoreClient()
    collection = client.get_collection_by_name(collection_name)

    vector_results = collection.search(
        natural_input=query,
        top_results=2,
        output_properties=[
            "title",
            "text"
        ]
    )

    contexts = [
        x.properties["text"] for x in vector_results
    ]

    vector_contexts = (
        "\n\n---\n\n".join(contexts)
    )
    return vector_contexts


def get_text() -> str:
    """
    Get the text from the user
    :return: string
    """
    input_text = st.text_input("You: ",
                               "Who is the biggest duck alive?",
                               key="input")
    return input_text


# From here down is all the StreamLit UI.
st.write("### FAQ Demo")

if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []


def main():
    """
    Streamlit app main method
    """

    chat_chain = llm_chain_response()
    user_input = get_text()
    if user_input:
        with st.spinner("Thinking..."):
            query = user_input
            vector_contexts = retrieve(query)
            prompt = f"{query} \nContext: {vector_contexts}"
            print("Prompt:")
            print(prompt)
            output = chat_chain.predict(input=prompt)
            print("Output:")
            print(output)
            st.session_state.past.append(user_input)
            st.session_state.generated.append(output)

    if st.session_state["generated"]:
        for i in range(len(st.session_state["generated"]) - 1, -1, -1):
            message(st.session_state["generated"][i], key=str(i))
            message(st.session_state["past"][i], is_user=True, key=str(i) + "_user", avatar_style="shapes")


if __name__ == "__main__":
    main()