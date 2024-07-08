from langchain_core.prompts import ChatPromptTemplate
from qwak import QwakClient
from qwak.llmops.prompt.manager import PromptManager
from langchain_openai import ChatOpenAI


def create_chain(prompt_name: str, model_name: str, model_url: str):
    prompt_manager = PromptManager()

    # Fetch the prompt from Qwak
    qwak_prompt = prompt_manager.get_prompt(
        name=prompt_name
    )

    # Convert to a Langchain template
    langchain_prompt = ChatPromptTemplate.from_messages(
        qwak_prompt.prompt.template.to_messages()
    )

    # Setup a LangChain LLM integration using the Qwak prompt configuration
    qwak_client = QwakClient()

    llm = ChatOpenAI(
        model=model_name,
        openai_api_base=model_url,
        openai_api_key=qwak_client.get_token(),
    )

    chain = langchain_prompt | llm

    # Invoke the chain with an optional variable
    return chain.invoke({"question": "What's your name?"})


if __name__ == '__main__':

    response = create_chain(
        prompt_name="banker-agent",
        model_name="llama_3_8b_instruct",
        model_url="https://models.qwak-prod.qwak.ai/v1/llama_3_8b_instruct"
    )
    print(response)

