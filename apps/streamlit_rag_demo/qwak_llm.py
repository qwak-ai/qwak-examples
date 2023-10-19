import logging
from typing import Any, Dict, List, Mapping, Optional

import pandas as pd
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from langchain.llms.utils import enforce_stop_tokens
from langchain.pydantic_v1 import Extra, Field, root_validator
from qwak_inference import RealTimeClient

logger = logging.getLogger(__name__)


class Qwak(LLM):
    """Qwak large language models.

    To use, you should have the ``qwak-inference`` python package installed.

    Any parameters that are valid to be passed to the call can be passed
    in, even if not explicitly saved on this class.

    Example:
        .. code-block:: python

            from langchain.llms import QwakLLM
            modal = QwakLLM(model_id="")

    """

    model_id: str = ""
    """Qwak model id to use"""

    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Holds any model parameters valid for `create` call not
    explicitly specified."""

    class Config:
        """Configuration for this pydantic config."""

        extra = Extra.forbid

    @root_validator(pre=True)
    def build_extra(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Build extra kwargs from additional params that were passed in."""
        all_required_field_names = {field.alias for field in cls.__fields__.values()}

        extra = values.get("model_kwargs", {})
        for field_name in list(values):
            if field_name not in all_required_field_names:
                if field_name in extra:
                    raise ValueError(f"Found {field_name} supplied twice.")
                logger.warning(
                    f"""{field_name} was transferred to model_kwargs.
                    Please confirm that {field_name} is what you intended."""
                )
                extra[field_name] = values.pop(field_name)
        values["model_kwargs"] = extra
        return values

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
            **{"model_id": self.model_id},
            **{"model_kwargs": self.model_kwargs},
        }

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "qwak"

    def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> str:
        """Call to Qwak RealTime model"""
        params = self.model_kwargs or {}
        params = {**params, **kwargs}

        columns = ["prompt"]
        data = [[prompt]]
        input_ = pd.DataFrame(data, columns=columns)
        # TODO - Replace this with a POST request so we don't import RealTimeClient
        client = RealTimeClient(model_id=self.model_id)

        response = client.predict(input_)
        try:
            text = response[0]["generated_text"][0]
        except KeyError:
            raise ValueError("LangChain requires 'generated_text' key in response.")

        if stop is not None:
            text = enforce_stop_tokens(text, stop)
        return text
