from openai import OpenAI
from qwak import QwakClient


def generate_response():
    qwak_client = QwakClient()

    client = OpenAI(
        base_url="https://models.qwak-prod.qwak.ai/v1/llama_3_8b_instruct",
        api_key=qwak_client.get_token()
    )

    chat_completion = client.chat.completions.create(
        model="llama_3_8b_instruct",
        messages=[
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "I'm a helpful assistant."},
            {"role": "user", "content": "Give me 7 types of fruits"}
        ]
    )

    print(chat_completion.choices[0].message.content)


if __name__ == '__main__':
    generate_response()
