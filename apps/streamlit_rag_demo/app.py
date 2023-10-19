
from chain import llm_chain_response
from vector_store import retrieve_vector_context

QWAK_MODEL_ID = 'llama2'

import streamlit as st
from streamlit_chat import message


def get_text() -> str:
    """
    Get the text from the user
    :return: string
    """
    input_text = st.text_input(label="You: ",
                               # value="",
                               key="input")
    return input_text


def extract_answer(text):
    split_text = text.split("Answer:")
    return split_text[-1]


# StreamLit UI
st.write("### FAQ Demo")

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

    use_content = st.checkbox(label="Use Vector Context", key="context-input")
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

            print(f"Prompt:\n {query}")
            print(f"Context:\n {context}")
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
