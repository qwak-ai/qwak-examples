from time import sleep

from qwak.llmops.model.descriptor import OpenAIChat
from qwak.llmops.prompt.manager import PromptManager


def main(prompt_name: str):
    prompt_manager = PromptManager()

    prompt = prompt_manager.get_prompt(
        name=prompt_name
    )

    while True:
        # Getting the default prompt version after several seconds
        # When the default version changes, the loaded prompt will change as well
        response = prompt.invoke(
            variables={
                "product_name": "Qwak",
                "product_type": "AI Infra Platform",
                "question": "what is Qwak?"
            },
            stream=True
            # model_override=OpenAIChat(
            #         model_id="gpt-4-0613",
            #         temperature=0.9,
            #         max_tokens=200
            # )
        )

        for chunk in response:
            print(chunk.choices[0].delta.content)

        # print(response.choices[0].message.content)
        sleep(10)
        print()


if __name__ == '__main__':

    prompt_name = "product-description-for-e-commerce"
    main(prompt_name=prompt_name)
