from qwak.inference.clients import RealTimeClient

# Please update MODEL_ID with your deployed model id
MODEL_ID = 'catboost'


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

  client = RealTimeClient(model_id="catboost")
  response = client.predict(feature_vector)
  print(response)
