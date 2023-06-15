from qwak_inference import RealTimeClient

# Please update MODEL_ID with your model id on Qwak
QWAK_MODEL_ID = 'flan_t5_fine_tuned'

if __name__ == '__main__':

    for i in range(100):
        input_ = [{
            "prompt": "Reporters at the Wall Street Journal interviewed the Russian soldier at a detention facility in the Kharkiv region on May 19. According to the Wall Street Journal, the soldier spoke while under the supervision of a guard. CNN cannot verify whether the soldier spoke under duress."
        }]
        client = RealTimeClient(model_id=QWAK_MODEL_ID)
        response = client.predict(input_)
        print(response)
