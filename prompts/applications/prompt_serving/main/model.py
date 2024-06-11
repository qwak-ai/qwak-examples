import pandas as pd
import qwak
from qwak.llmops.prompt.manager import PromptManager
from qwak.model.base import QwakModel
from qwak.model.schema import ExplicitFeature, ModelSchema


class PromptApplication(QwakModel):

    def __init__(self):
        self.prompt_manager = None
        self.prompt = None
        self.default_prompt_name = "banker-agent"

    def build(self):
        pass

    # Define the input schema for the model
    def schema(self):
        model_schema = ModelSchema(
            inputs=[
                ExplicitFeature(name="prompt", type=str),
            ])
        return model_schema

    def initialize_model(self):
        self.prompt_manager = PromptManager()
        # We get the prompt here once to start hot-loading
        # The prompt object will automatically be updated
        # with newer default prompt versions
        self.prompt = self.prompt_manager.get_prompt(
            name=self.default_prompt_name
        )

    def _get_prompt(self, name: str):
        return self.prompt_manager.get_prompt(
            name=name
        )

    @qwak.api()
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        user_input_prompt = df['prompt'][0]
        prompt_name = df['prompt_name'][0] if 'prompt_name' in df.columns else None

        # Flatting the dataframe
        variables = df.to_dict(orient='records')[0]
        variables.pop('prompt_name', None)
        # Setting the prompt as the question variable
        variables["question"] = variables["prompt"]

        if not user_input_prompt:
            return pd.DataFrame([])

        # Determine which prompt to use, so we keep hot-loading the default prompt
        if not prompt_name or (prompt_name == self.default_prompt_name):
            prompt = self.prompt
        else:
            prompt = self._get_prompt(name=prompt_name)

        response = prompt.invoke(variables=variables)

        return pd.DataFrame([{"generated_text": response.choices[0].message.content}])