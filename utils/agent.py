from typing import List, Union, Set, Dict, Any, AsyncGenerator
from pydantic import BaseModel, Field 
from pathlib import Path
import asyncio  
import aiosqlite
import json 
import tiktoken  # 新增：导入tiktoken用于手动计算Token

def patch_aiosqlite_is_alive():
    """Patch for aiosqlite Connection missing is_alive method"""
    if not hasattr(aiosqlite.Connection, 'is_alive'):
        def is_alive(self):
            """Return True if the underlying sqlite3 connection exists."""
            return self._connection is not None
        aiosqlite.Connection.is_alive = is_alive
patch_aiosqlite_is_alive()

# 初始化Token编码器（兼容通义千问/OpenAI）
encoding = tiktoken.get_encoding("cl100k_base")

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

        tool_call_ids: Set[str] = set()  # Track called tool IDs
        total_input_tokens: int = 0  # Total input tokens (including function call message)
        total_output_tokens: int = 0  # Total output tokens
        usage: Dict[str, int] | None = None  # Token usage statistics
        full_output_content = ""  # Store complete AI response content
        tool_call_content = ""  # Store tool call description text for token calculation
        tool_result_content = ""  # Store tool response text for token calculation
        
        # Extract agent's config from `AgentConfig`
        model = self.agent_config.model
        tools = self.agent_config.tools
        memory_uri = self.agent_config.memory_uri_str  
        system_prompt = self.agent_config.system_prompt
        
        # ========== 手动计算输入Token（用户提问） ==========
        total_input_tokens += len(encoding.encode(query))
        # 系统提示词也计入输入Token（可选，根据需求调整）
        total_input_tokens += len(encoding.encode(system_prompt))
        
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
                # 移除原有依赖usage_metadata的Token统计逻辑
                
                # Function call message
                if isinstance(message, AIMessage) and message.tool_calls:
                    for tool_call in message.tool_calls: 
                        tool_call_id = tool_call.get("id")
                        tool_name = tool_call.get("name")
                        tool_args = tool_call.get("args")
                        # 累加工具调用描述文本（用于计算Token）
                        tool_call_text = f"{tool_name}{json.dumps(tool_args)}"
                        tool_call_content += tool_call_text
                        
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
            
                # Completion message (AI text response)
                elif isinstance(message, AIMessage) and message.content: 
                    # 累加完整回复内容，用于计算输出Token
                    full_output_content += message.content
                    for char in message.content: 
                        yield {
                            "type": "content", 
                            "content": char
                        }
            
                # Tool response message
                elif isinstance(message, ToolMessage):
                    called_tool_name = message.name 
                    called_tool_content = message.content # Parse for detailed tool response info
                    # 累加工具返回结果文本（用于计算Token）
                    tool_result_content += called_tool_content or ""
                    
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
            
            # ========== 最终Token统计（完全手动计算） ==========
            # 工具调用/返回的Token计入输入Token
            tool_call_tokens = len(encoding.encode(tool_call_content))
            tool_result_tokens = len(encoding.encode(tool_result_content))
            total_input_tokens += tool_call_tokens + tool_result_tokens
            
            # AI回复的Token计入输出Token
            total_output_tokens = len(encoding.encode(full_output_content))
            
            # 总Token计算
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