USER_ID = "test_user"  # TODO 实际的user_id要由网关层/后端传入 

from datetime import datetime
from pathlib import Path 
import uuid 
import json
from fastapi import FastAPI, HTTPException, UploadFile, Form
from fastapi.responses import StreamingResponse 
from pydantic import BaseModel, Field 
import uvicorn
from typing import Dict, Any, Optional, List, Union


from utils.llms import model 
from utils.tools import tools
from utils.prompt import SYSTEM_PROMPT as system_prompt
from utils.agent import AgentConfig, AgentService


class ChatRequest(BaseModel): 
    user_id: str = Field(description="user id", min_length=1, max_length=64)
    session_id: Optional[str] = Field(None, description="will be set automatically if none")
    query: str = Field(description="user's question", min_length=1, max_length=2000)

class ChatResponse(BaseModel): 
    type: str = Field(description="content/tool_call/tool_result/usage/error") 
    content: str | Dict = Field(description="content")

agent_config = AgentConfig(
    model=model, 
    tools=tools, 
    memory_uri="",  
    system_prompt=system_prompt
)
agent_service = AgentService(agent_config)

app = FastAPI(
    title="ChatAgent API Server", 
    version="v1.0.0"
)

@app.get("/")
async def root(): 
    return {
        "message": "ChatAgent API Server is running!",
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

@app.get("/health")
async def health(): 
    return {
        "status": "healthy", 
        "service": "chat-agent-api",
        "version": "1.0.0",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "message": "All services are running normally"
    }

@app.post("/v1/chat/completions", summary="streaming response api")
async def stream_chat(request: ChatRequest):
    try: 
        session_id = request.session_id or str(uuid.uuid4())
        memory_dir = Path("./memories") / request.user_id
        memory_dir.mkdir(parents=True, exist_ok=True)
        memory_uri = memory_dir / f"session_{session_id}.db"
        agent_service.agent_config.memory_uri = memory_uri
        
        async def response_generator():
            try:
                async for resp in agent_service.call_agent(
                    query=request.query,
                    session_id=session_id
                ):
                    yield f"data: {json.dumps(resp, ensure_ascii=False)}\n\n"
            except Exception as e:
                error_resp = {
                    "type": "error",
                    "content": f"调用Agent失败: {str(e)}"
                }
                yield f"data: {json.dumps(error_resp, ensure_ascii=False)}\n\n"
        
        return StreamingResponse(
            response_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no" 
            }
        )
    except HTTPException:
        raise  
    except Exception as e: 
        print(f"Server Error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Server Error: {str(e)}"
        )

if __name__ == "__main__": 
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000, 
        log_level="info",
        reload=False
    )