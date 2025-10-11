from typing import TypedDict, Annotated, List, Union
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.messages import BaseMessage
from dataclasses import dataclass, field
import operator

@dataclass
class AgentState:
    input: str | None = None
    chat_history: List[BaseMessage] = field(default_factory=list)
    intermediate_steps: Annotated[List[tuple[AgentAction, str]], operator.add] = field(default_factory=list)
    output: str | None = None

@dataclass
class InputState:
    input: str | None = None

@dataclass
class OutputState:
    output: str | None = None
    #intermediate_steps: Annotated[List[tuple[AgentAction, str]], operator.add] = field(default_factory=list)