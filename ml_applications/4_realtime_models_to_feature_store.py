from qwak import QwakClient
from qwak.applications import QwakApplication, step
from qwak_inference import RealTimeClient


class RealTimeModelToFeatureStoreApp(QwakApplication):
    """
    1. Run the real time model
    2. Save the model output in the feature store
    """
    @step
    def start(self):
        self.next(self.run_realtime_model)

    @step
    def run_realtime_model(self, params):
        df = params["df"]
        client = RealTimeClient(model_id="sentence_embeddings_aecc6f")
        flan_output = client.predict(df)
        self.next(self.run_batch_feature_ingestion,
                  params={
                      "df": flan_output
                  })

    @step
    def run_batch_feature_ingestion(self):
        # 1. Use the QwakClient and wait for completion
        client = QwakClient()
        client.trigger_batch_feature_set("user-credit-risk-features")
        self.next(self.perform_batch_feature_ingestion)

        # 2. Have the application perform it natively
        self.execute_batch_feature_set("user-features",
                                       on_complete=self.run_batch_model_inference,
                                       execution_params={})


if __name__ == '__main__':
    flow = RealTimeModelToFeatureStoreApp()
    flow.execute(schedule="0 * * * *")
    flow.execute_now()
