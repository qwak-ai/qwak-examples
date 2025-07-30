from typing import Any, Generator, Optional, List, Dict
from mlflow.pyfunc import ChatAgent
from mlflow.types.agent import ChatAgentChunk, ChatAgentMessage, ChatAgentResponse, ChatContext
from langgraph.graph.state import CompiledStateGraph


class LangGraphChatAgent(ChatAgent):
    def __init__(self, agent: CompiledStateGraph):
        self.agent = agent

    def predict(
        self, 
        messages: List[ChatAgentMessage], 
        context: Optional[ChatContext] = None, 
        custom_inputs: Optional[Dict[str, Any]] = None
    ) -> ChatAgentResponse:
        request = {"messages": self._convert_messages_to_dict(messages)}
        messages = []
        for event in self.agent.stream(request, stream_mode="updates"):
            for node_data in event.values():
                messages.extend(
                    ChatAgentMessage(**msg) for msg in node_data.get("messages", [])
                )
        return ChatAgentResponse(messages=messages)
    
    def predict_stream(
        self, 
        messages: List[ChatAgentMessage], 
        context: Optional[ChatContext] = None, 
        custom_inputs: Optional[Dict[str, Any]] = None
    ) -> Generator[ChatAgentChunk, None, None]:
        request = {"messages": self._convert_messages_to_dict(messages)}
        for event in self.agent.stream(request, stream_mode="updates"):
            for node_data in event.values():
                yield from (
                    ChatAgentChunk(**{"delta": msg}) for msg in node_data["messages"]
                )