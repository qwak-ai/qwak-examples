from qwak.llmops.model.descriptor import OpenAIChat
from qwak.llmops.prompt.base import ChatPrompt
from qwak.llmops.prompt.chat.template import HumanMessagePromptTemplate, SystemMessagePromptTemplate, ChatPromptTemplate
from qwak.llmops.prompt.manager import PromptManager


def manage_prompts(name: str,
                   description: str,
                   prompt_manager: PromptManager):
    chat_template = ChatPromptTemplate(
        messages=[
            SystemMessagePromptTemplate("You are a friendly and knowledgeable AI assistant."
                                        " Provide clear, accurate, and concise information, and if unsure, admit it."),
            HumanMessagePromptTemplate("""Please create a query that represents this request from an analyst:
{{query}}

Use the following SQL queries as examples:
SELECT name, email 
FROM employees 
WHERE hire_date > '2020-01-01' 
AND department = 'Marketing';"""),
        ],
    )

    model = OpenAIChat(
        model_id="gpt-4o-mini",
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
        version_description="Initial prompt version"
    )


if __name__ == '__main__':
    manager = PromptManager()

    manage_prompts(
        name='query-translator',
        description='Converting user requests to SQL queries',
        prompt_manager=manager
    )
