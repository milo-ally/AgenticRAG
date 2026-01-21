from langchain.tools import tool
from pydantic import BaseModel, Field
import requests
import json
import random

class WeatherArgs(BaseModel):
    location: str = Field(description="the name of the city")

class SemanticSearchArgs(BaseModel):
    query: str = Field(description="Query statement for semantic search")
    kb_id: str = Field(description="Knowledge base ID (optional if filename is provided)", default="")
    filename: str = Field(description="Filename to lookup knowledge base (optional if kb_id is provided)", default="")
    top_k: int = Field(description="Number of most relevant results to return", default=3)
    delete_immediate: bool = Field(description="Whether to delete the knowledge base immediately after search", default=False)

# 示例工具：获取天气信息(mock)
@tool(args_schema=WeatherArgs)
def get_weather(
    location: str
) -> str:
    """ 
    Args: 
        location: the name of the city
    Returns: 
        the temperature of the city
    """
    weather_conditions = ["Sunny", "Cloudy", "Rainy", "Windy", "Snowy"]
    temperature = random.randint(10, 30)
    return json.dumps({
        "content": f"Weather in {location}: {random.choice(weather_conditions)}, Temperature: {temperature}°C",
        "metadata": {
            "tokens": 111, 
            "shit": True
        }
    }, ensure_ascii=False)

@tool(args_schema=SemanticSearchArgs)
def semantic_search(
    query: str, 
    kb_id: str = "",
    filename: str = "",
    top_k: int = 3, 
    delete_immediate: bool = True,
) -> str:
    """
    Call RAG service to perform semantic search, return relevant context and token usage.
    Can search by knowledge base ID or filename. If filename is provided, it will be used to lookup the kb_id.
    
    Args: 
        query: Query statement for semantic search
        kb_id: Knowledge base ID (optional if filename is provided)
        filename: Filename to lookup knowledge base (optional if kb_id is provided)
        top_k: Number of most relevant results to return
        delete_immediate: Whether to delete the knowledge base immediately after search
        
    Returns: 
        JSON string containing two fields:
        - content: Search results (context) or error information
        - metadata: Additional information including token usage, status code, etc.
    """
    result_dict = None
    actual_kb_id = None
    
    try:
        rag_service_base_url = "http://localhost:5000"
        rag_service_base_url = rag_service_base_url.rstrip()
        
        # 如果提供了文件名但没有kb_id，则通过文件名查找kb_id
        if filename and not kb_id:
            get_kb_id_url = f"{rag_service_base_url}/get-kb-id-by-filename"
            kb_id_response = requests.get(
                get_kb_id_url,
                params={"filename": filename},
                timeout=30
            )
            if kb_id_response.status_code == 200:
                actual_kb_id = kb_id_response.json()["data"]["knowledge_base_id"]
            else:
                return json.dumps({
                    "content": f"Failed to find knowledge base for filename '{filename}': {kb_id_response.text}",
                    "metadata": {
                        "token_usage": 0,
                        "status_code": kb_id_response.status_code,
                        "filename": filename,
                        "error": "Knowledge base not found"
                    }
                }, ensure_ascii=False)
        else:
            actual_kb_id = kb_id
        
        if not actual_kb_id:
            return json.dumps({
                "content": "Either kb_id or filename must be provided",
                "metadata": {
                    "token_usage": 0,
                    "status_code": 400,
                    "error": "Missing kb_id or filename"
                }
            }, ensure_ascii=False)
        
        search_url = f"{rag_service_base_url}/search"

        payload = {
            "knowledge_base_id": actual_kb_id,
            "query": query,
            "top_k": top_k
        }

        search_response = requests.post(
            search_url,
            json=payload,
            timeout=30  # Add timeout setting
        )
        
        if search_response.status_code == 200: 
            content = search_response.json()
            token_usage = content["data"]["token_stats"]
            results = content["data"]["results"]
            context = "\n\n".join([result["content"] for result in results])
            result_dict = {
                "content": context,
                "metadata": {
                    "token_usage": token_usage,
                    "status_code": 200,
                    "kb_id": actual_kb_id,
                    "filename": filename if filename else None,
                    "top_k": top_k
                }
            }
        else:
            result_dict = {
                "content": f"retrieval failed! status code: {search_response.status_code}, reason: {search_response.text}",
                "metadata": {
                    "token_usage": 0,
                    "status_code": search_response.status_code,
                    "kb_id": actual_kb_id,
                    "filename": filename if filename else None
                }
            }
    except Exception as e:
        result_dict = {
            "content": f"retrieval failed! error: {str(e)}",
            "metadata": {
                "token_usage": 0,
                "kb_id": actual_kb_id if actual_kb_id else None,
                "filename": filename if filename else None,
                "error": str(e)
            }
        }
    finally:
        if delete_immediate and actual_kb_id and result_dict:
            delete_url = f"{rag_service_base_url.rstrip('/')}/delete-knowledge-base"
            delete_params = {"knowledge_base_id": actual_kb_id}
            try:
                requests.delete(delete_url, params=delete_params, timeout=30)
                if result_dict and "metadata" in result_dict:
                    result_dict["metadata"]["delete_immediate"] = True
            except Exception as e:
                if result_dict and "metadata" in result_dict:
                    result_dict["metadata"]["delete_immediate"] = False
                    result_dict["metadata"]["delete_error"] = str(e)
    
    return json.dumps(result_dict, ensure_ascii=False)

tools = [get_weather, semantic_search]