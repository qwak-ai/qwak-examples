import streamlit as st
from streamlit_chat import message

from chain import llm_chain_response
from vector_store import retrieve_vector_context

QWAK_MODEL_ID = 'flan_t5'


def get_text() -> str:
    """
    Get the text from the user
    :return: string
    """
    input_text = st.text_input(label="You: ",
                               value="Who is the biggest duck alive?",
                               key="input")
    return input_text


# StreamLit UI.
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

    user_input = get_text()
    if user_input:
        with st.spinner("Thinking..."):
            query = user_input
            vector_contexts = retrieve_vector_context(query=query,
                                                      output_properties=[
                                                          "title",
                                                          "text"
                                                      ])

            prompt = f"{query} \nContext: {vector_contexts}"
            print(f"Prompt:\n {prompt}")

            output = chat_chain.predict(input=prompt)
            print(f"Output:\n{output}")

            # Add the responses to the chat state
            st.session_state.past.append(user_input)
            st.session_state.generated.append(output)

    if st.session_state["generated"]:
        for i in range(len(st.session_state["generated"]) - 1, -1, -1):
            message(st.session_state["generated"][i], key=str(i))
            message(st.session_state["past"][i], is_user=True, key=str(i) + "_user", avatar_style="shapes")


if __name__ == "__main__":
    main()
