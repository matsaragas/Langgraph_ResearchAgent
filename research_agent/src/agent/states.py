from typing import TypedDict, Annotated, List, Union
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.messages import BaseMessage
from dataclasses import dataclass, field
import operator

@dataclass
class AgentState:
    input: str = field(default=None)
    chat_history: list[BaseMessage] = field(default=None)
    intermediate_steps: Annotated[list[tuple[AgentAction, str]], operator.add] = field(default=None)

@dataclass
class InputState:
    input: str = field(default=None)

@dataclass
class OutputState:
    intermediate_steps: Annotated[list[tuple[AgentAction, str]], operator.add] = field(default=None)