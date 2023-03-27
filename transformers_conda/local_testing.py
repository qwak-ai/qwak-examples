from qwak.inference.clients import RealTimeClient

# Please update MODEL_ID with your deployed model id
MODEL_ID = 'guy_eshet_huggingface_model'

if __name__ == '__main__':
  feature_vector = [{
          'text': 'The best place ever!'
  }]

  client = RealTimeClient(model_id=MODEL_ID)
  response = client.predict(feature_vector)
  print(response)