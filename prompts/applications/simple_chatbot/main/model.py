import pandas as pd
import qwak
from qwak.llmops.prompt.manager import PromptManager
from qwak.model.base import QwakModel
from qwak.model.schema import ExplicitFeature, ModelSchema


class SimpleChatbot(QwakModel):

    # Initialize model parameters
    def __init__(self):
        self.prompt_manager = None
        self.prompt = None
        self.prompt_name = "banker-agent"

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
            name=self.prompt_name
        )

    @qwak.api()
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        # Getting the input prompt from the dataframe
        user_input_prompt = df['prompt'][0]

        if not user_input_prompt:
            return pd.DataFrame([])

        response = self.prompt.invoke(
            variables={"question": user_input_prompt}
        )

        return pd.DataFrame([
            {
                "generated_text": response.choices[0].message.content
            }
        ])
