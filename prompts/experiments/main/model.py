import pandas as pd
import qwak
from qwak import log_metric, log_data, log_file
from qwak.llmops.prompt.manager import PromptManager
from qwak.model.base import QwakModel
from qwak.model.experiment_tracking import log_params
from qwak.model.schema import ExplicitFeature, ModelSchema


class PromptExperimentsManager(QwakModel):

    # Initialize model parameters
    def __init__(self):
        self.prompt_manager = None
        self.prompt = None
        self.prompt_name = "query-translator"

    def build(self):
        self.prompt = PromptManager().get_prompt(
            name=self.prompt_name
        )
        log_metric({
            "prompt_version": self.prompt.version
        })
        log_params({
            "prompt_name": self.prompt.name
        })
        log_file(
            from_path="main/natural_language_to_sql.csv",
            tag="dataset_1"
        )

    # Define the input schema for the model
    def schema(self):
        model_schema = ModelSchema(
            inputs=[
                ExplicitFeature(name="prompt", type=str),
            ])
        return model_schema

    def initialize_model(self):
        # We get the prompt here once to start hot-loading
        # The prompt object will automatically be updated
        # with newer default prompt versions
        self.prompt_manager = PromptManager()
        self.prompt = self.prompt_manager.get_prompt(
            name=self.prompt_name
        )

    @qwak.api()
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:

        generated_text_list = []
        user_input_prompt = df['prompt']

        log_data(df, "received_dataset")

        for prompt in user_input_prompt:

            if not prompt:
                return pd.DataFrame([])

            response = self.prompt.invoke(
                variables={"query": prompt}
            )

            generated_text = response.choices[0].message.content
            generated_text_list.append(generated_text)

        return pd.DataFrame(
            {"generated_text": generated_text_list}
        )

