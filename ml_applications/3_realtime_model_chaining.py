from pandas import DataFrame
from qwak.applications import QwakRealTimeApplication, step
from qwak_inference import RealTimeClient


class RealTimeModelsChainingApp(QwakRealTimeApplication):
    """
    1. Run the real time model "flan_t5_5bdf83fa"
    2. Take the model output and run it through "sentence_embeddings_aecc6f"
    3. Return the output

    Open Issues:
    - How do we support multi-model deployment: Redeploying the whole flow when a model updates
    - Should we add a context?
    """

    @step
    def start(self, df):
        self.next(self.run_first_realtime_model,
                  params={
                      "df": df
                  })

    @step
    def run_first_realtime_model(self, params):
        df = params["df"]
        client = RealTimeClient(model_id="flan_t5_5bdf83fa")
        flan_output = client.predict(df)
        self.next(self.run_second_realtime_model,
                  params={
                      "df": flan_output
                  })

    @step
    def run_second_realtime_model(self, params):
        # [{"generated_text": ["love is a resemblance to love."]}]
        df = params["df"]
        new_df = DataFrame({"input": df["generated_text"]}, index=[0])

        client = RealTimeClient(model_id="sentence_embeddings_aecc6f")
        embeddings = client.predict(new_df)
        self.next(self.end,
                  params={
                      "df": embeddings
                  })


if __name__ == '__main__':
    flow = RealTimeModelsChainingApp()
    flow.deploy(name="flan_embeddings")

    feature_vector = [
        {
            "input": "hello world"
        }
    ]
    client = RealTimeClient(model_id="flan_embeddings")
    embeddings = client.predict(feature_vector)

    # Open questions
    # 1. How to connect two real time models without the PredictionClient
    # 2. Maybe use a QwakRealTimeApplication?
