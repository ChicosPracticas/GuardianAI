from openai import AsyncOpenAI
from openai.types.responses import EasyInputMessageParam
from openai import RateLimitError, APIConnectionError, APIStatusError
from typing import List, Literal, cast
from collections.abc import AsyncGenerator

from application.ports import LanguageModel
from domain.value_objects import Message
from domain.exceptions import ProviderAPIError, ProviderConnectionError, ProviderRateLimitError
from domain.value_objects.TokenUsage import TokenUsage


class OpenAILanguageModel(LanguageModel):
    def __init__(self, api_key: str):
        self.client = AsyncOpenAI(api_key=api_key)

    async def stream(self, messages: List[Message], model: str, usage: TokenUsage) -> AsyncGenerator[str, None]:
        input_messages: List[EasyInputMessageParam] = [
            EasyInputMessageParam(
                role=cast(Literal["user", "assistant", "system", "developer"], message.role.value),
                content=str(message.content)
            )
            for message in messages
        ]

        try:
            async with self.client.responses.stream(
                model=model,
                input=input_messages,
            ) as s:
                async for event in s:
                    if event.type == "response.output_text.delta":
                        yield event.delta
                    elif event.type == "response.completed":
                        usage.prompt_tokens = event.response.usage.input_tokens
                        usage.completion_tokens = event.response.usage.output_tokens

        except RateLimitError as e:
            raise ProviderRateLimitError(str(e))
        except APIConnectionError as e:
            raise ProviderConnectionError(str(e))
        except APIStatusError as e:
            raise ProviderAPIError(str(e))