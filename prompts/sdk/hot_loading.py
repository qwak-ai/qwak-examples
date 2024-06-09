from time import sleep
from qwak.llmops.prompt.manager import PromptManager


def main(prompt_name: str):
    prompt_manager = PromptManager()

    bank_agent_prompt = prompt_manager.get_prompt(
        name=prompt_name
    )

    while True:
        # Getting the default prompt version after several seconds
        # When the default version changes, the loaded prompt will change as well
        response = bank_agent_prompt.invoke(
            variables={
                "question": "Tell me a joke please"
            }
        )

        print(response.choices[0].message.content)
        sleep(10)
        print()


if __name__ == '__main__':

    prompt_name = "banker-agent"
    main(prompt_name=prompt_name)
