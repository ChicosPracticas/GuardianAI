from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator

from domain.value_objects.TokenUsage import TokenUsage
from domain.value_objects.message import Message


class LanguageModel(ABC):

    @abstractmethod
    async def stream(self, messages: list[Message], model: str, usage: TokenUsage) -> AsyncGenerator[str, None]:
        ...