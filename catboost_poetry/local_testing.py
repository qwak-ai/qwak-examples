from qwak_inference import RealTimeClient

# Please update MODEL_ID with your model id on Qwak
QWAK_MODEL_ID = 'your-model-id'


if __name__ == '__main__':
  feature_vector = [
    {
      "UserId" : "male",
      "Age" : 3,
      "Sex" : "male",
      "Job" : 2,
      "Housing" : "male",
      "Saving accounts" : "male",
      "Checking account" : "male",
      "Credit amount" : 54.2,
      "Duration" : 4,
      "Purpose" : "male",
      "Age_cat" : "male",
      
    }]

  client = RealTimeClient(model_id=QWAK_MODEL_ID)
  response = client.predict(feature_vector)
  print(response)
