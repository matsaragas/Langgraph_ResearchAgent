from typing import TypedDict, Annotated, List, Union
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.messages import BaseMessage
import operator

class AgentState:
    input: str = field(default=None)
    chat_history: list[BaseMessage] = field(default=None)
    itermediate_steps: Annotated[list[tuple[AgentAction, str]], operator.add]

class InputState:
    input: str = field(default=None)

class OutputState:
    itermediate_steps: Annotated[list[tuple[AgentAction, str]], operator.add]