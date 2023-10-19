import streamlit as st
from streamlit_chat import message

from chain import llm_chain_response
from vector_store import retrieve_vector_context

QWAK_MODEL_ID = 'llama2'


def get_text() -> str:
    """
    Get the text from the user
    :return: string
    """
    input_text = st.text_input(label="You: ",
                               key="input")
    return input_text


def extract_answer(text):
    split_text = text.split("Answer:")
    return split_text[-1]


# StreamLit UI
left_co, cent_co, last_co = st.columns(3)
with cent_co:
    image = open("qwak.png", "rb").read()
    st.image(image, use_column_width="auto")

st.write("### Vector Store RAG Demo")


if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []


def main():
    """
    Streamlit app main method
    """

    # Get a new chat chain to query LLMs
    chat_chain = llm_chain_response(model_id=QWAK_MODEL_ID)

    use_content = st.checkbox(label="Use Vector Store", key="use-vector-store")
    user_input = get_text()

    if user_input:
        with st.spinner("Thinking..."):
            query = user_input
            if use_content:
                context = retrieve_vector_context(query=query,
                                                  collection_name="financial-data",
                                                  output_key='chunk_text',
                                                  top_results=4,
                                                  properties=[
                                                      "chunk_text"
                                                  ])
            else:
                context = ""

            print(f"Prompt:\n{query}")
            print(f"Context:\n{context}")
            output = chat_chain({
                "input": query,
                "context": context
            })
            answer = extract_answer(output["text"])

            print(f"Output:\n{answer}")

            # Add the responses to the chat state
            st.session_state.past.append(user_input)
            st.session_state.generated.append(answer)

    if st.session_state["generated"]:
        for i in range(len(st.session_state["generated"]) - 1, -1, -1):
            message(st.session_state["generated"][i], key=str(i))
            message(st.session_state["past"][i], is_user=True, key=str(i) + "_user", avatar_style="shapes")


if __name__ == "__main__":
    main()
