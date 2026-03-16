"""Patches for SDK compatibility."""

from litellm.completion_extras.litellm_responses_transformation.transformation import (
    LiteLLMResponsesTransformationHandler,
)
from openhands.sdk.llm.message import TextContent

original_convert = LiteLLMResponsesTransformationHandler._convert_content_str_to_input_text


def patched_convert(self, content, role):
    return {"type": "input_text", "text": content}


LiteLLMResponsesTransformationHandler._convert_content_str_to_input_text = patched_convert

original_model_dump = TextContent.model_dump


def patched_model_dump(self, **kwargs):
    result = original_model_dump(self, **kwargs)
    result.pop("enable_truncation", None)
    return result


TextContent.model_dump = patched_model_dump
