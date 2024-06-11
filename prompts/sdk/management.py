from qwak.llmops.model.descriptor import OpenAIChat
from qwak.llmops.prompt.base import ChatPrompt
from qwak.llmops.prompt.manager import PromptManager
from qwak.llmops.prompt.chat.template import SystemMessagePromptTemplate, AIMessagePromptTemplate, \
    HumanMessagePromptTemplate, ChatPromptTemplate


def manage_prompts(name: str,
                   description: str,
                   prompt_manager: PromptManager):

    chat_template = ChatPromptTemplate(
        messages=[
            SystemMessagePromptTemplate("You are a funny and polite banking assistant"),
            AIMessagePromptTemplate("How can I help?"),
            HumanMessagePromptTemplate("{{question}}"),
        ],
    )

    model = OpenAIChat(
        model_id="gpt-3.5-turbo",
        temperature=0.9
    )
    chat_prompt = ChatPrompt(
        template=chat_template,
        model=model
    )

    return prompt_manager.register(
        name=name,
        prompt=chat_prompt,
        prompt_description=description,
        version_description="Initial prompt"
    )


def generate_response(name: str, prompt_manager: PromptManager):

    bank_agent_prompt = prompt_manager.get_prompt(
        name=name
    )

    return bank_agent_prompt.invoke(
        variables={
            "question": "Tell me a joke please"
        }
    )


if __name__ == '__main__':
    prompt_name = "banker-agent"
    description = "Testing a banker agent prompt"

    # Create an instance of the prompt manager
    prompt_manager = PromptManager()

    # Register a new prompt
    prompt = manage_prompts(name=prompt_name,
                            description=description,
                            prompt_manager=prompt_manager)

    # Generate a response from the model
    response = generate_response(name=prompt_name, prompt_manager=prompt_manager)
    print(response.choices[0].message.content)

    # Deleting the prompt
    prompt_manager.delete_prompt(
        name=prompt.name
    )