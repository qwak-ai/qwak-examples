from time import sleep

from inference import flan_completion, FLAN_T5_FINETUNED_MODEL_ID
from faker import Faker

if __name__ == '__main__':
    while True:
        try:
            fake = Faker()
            sentence = fake.paragraph(nb_sentences=1)
            print(sentence)
            response = flan_completion(
                input_text=sentence,
                model_id=FLAN_T5_FINETUNED_MODEL_ID
            )
            print(f"Response: {response}")
            sleep(1)
        except Exception as e:
            print(f"Exception: {e}")
