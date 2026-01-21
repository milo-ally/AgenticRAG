from typing import List, Union, Set, Dict, Any, AsyncGenerator
from pydantic import BaseModel, Field 
from pathlib import Path
import asyncio  
import aiosqlite
import json 

def patch_aiosqlite_is_alive():
    """补丁"""
    if not hasattr(aiosqlite.Connection, 'is_alive'):
        def is_alive(self):
            """Return True if the underlying sqlite3 connection exists."""
            return self._connection is not None
        aiosqlite.Connection.is_alive = is_alive
patch_aiosqlite_is_alive()

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langchain.agents import create_agent
from langchain.chat_models import BaseChatModel
from langchain.tools import BaseTool
from langchain_core.messages import AIMessage, ToolMessage
from typing import Optional


class AgentConfig(BaseModel):  
    model: BaseChatModel = Field(..., description="Such as ChatOpenAI")
    tools: List[BaseTool] = Field(default_factory=list, description="List of callable langchain tools")
    memory_uri: Union[str, Path] = Field(..., description="The Path to SQLite Database to memory storage")
    system_prompt: str = Field(..., description="system_prompt")
    
    class Config:
        arbitrary_types_allowed = True
    @property
    def memory_uri_str(self) -> str:
        return str(self.memory_uri)
    

class AgentService:
    def __init__(self, agent_config: AgentConfig):
        self.agent_config = agent_config  
    
    async def call_agent(
        self,
        query: str,  # user query
        session_id: str,  # 1 session-id <-> 1 memory
    ) -> AsyncGenerator[Dict[str, Any], None]:

        tool_call_ids: Set[str] = set()  # 统计调用的工具id
        total_input_tokens: int = 0  # total_input_tokens (including function call message)
        total_output_tokens: int = 0  # total_output_tokens
        usage: Dict[str, int] | None = None  # token usage
        
        # extract agent's config from `AgentConfig`
        model = self.agent_config.model
        tools = self.agent_config.tools
        memory_uri = self.agent_config.memory_uri_str  
        system_prompt = self.agent_config.system_prompt
        
        async with AsyncSqliteSaver.from_conn_string(memory_uri) as checkpointer:
            await checkpointer.setup()
            agent = create_agent(
                model=model, 
                tools=tools, 
                checkpointer=checkpointer, 
                system_prompt=system_prompt
            )
            config = {
                "configurable": {
                    "thread_id": session_id
                }
            }
            messages = [
                {
                    "role": "user", 
                    "content": query
                }
            ]
            async for chunk in agent.astream(
                {"messages": messages}, 
                config=config, 
                stream_mode="messages"
            ):
                message = chunk[0]
                state = await agent.aget_state(config=config)

                # starting token analyzation
                tasks = state.tasks
                for task in tasks:
                    if task.result is not None: 
                        for msg in task.result["messages"]:
                            if isinstance(msg, AIMessage) and hasattr(msg, 'usage_metadata'):
                                usage_metadata = msg.usage_metadata
                                current_input = usage_metadata.get("input_tokens", 0)
                                current_output = usage_metadata.get("output_tokens", 0)
                                if current_input not in [0, total_input_tokens]:
                                    total_input_tokens += current_input
                                if current_output not in [0, total_output_tokens]:
                                    total_output_tokens += current_output
                
                # function call message
                if isinstance(message, AIMessage) and message.tool_calls:
                    for tool_call in message.tool_calls: 
                        tool_call_id = tool_call.get("id")
                        tool_name = tool_call.get("name")
                        tool_args = tool_call.get("args")
                        if tool_name and tool_call_id and tool_call_id not in tool_call_ids:
                            yield {
                                "type": "tool_call", 
                                "content": {
                                    "tool_name": tool_name, 
                                    "tool_call_id": tool_call_id, 
                                    "tool_args": tool_args
                                }
                            }
                            tool_call_ids.add(tool_call_id)
            
                # completion message
                elif isinstance(message, AIMessage) and message.content: 
                    for char in message.content: 
                        yield {
                            "type": "content", 
                            "content": char
                        }
            
                # tool response message
                elif isinstance(message, ToolMessage):
                    called_tool_name = message.name 
                    called_tool_content = message.content # 可以解析这个字段来获取工具函数返回的详细信息
                    try:
                        called_tool_metadata = str(json.loads(called_tool_content).get("metadata", {})) \
                        if called_tool_content else "{}"
                    except (json.JSONDecodeError, TypeError):
                        called_tool_metadata = "{}"
                    yield {
                        "type": "tool_result", 
                        "content": {
                            "called_tool_name": called_tool_name, 
                            "called_tool_content": called_tool_content, 
                            "called_tool_metadata": called_tool_metadata,
                        }
                    }
            
            # yield token statisitics
            total_tokens = total_input_tokens + total_output_tokens
            usage = {
                "input_tokens": total_input_tokens, 
                "output_tokens": total_output_tokens, 
                "total_tokens": total_tokens
            }
            yield {
                "type": "usage",
                "content": usage
            }