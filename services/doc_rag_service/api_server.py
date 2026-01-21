"""
PDF AgenticRAG Semantic Search API
Functionality: Upload PDF files and build vector databases, support semantic retrieval, include Token consumption statistics
Dependencies: langchain langchain-community faiss-cpu pypdfium2 tiktoken dashscope fastapi uvicorn pydantic numpy redis
"""
from langchain_core.documents.base import Blob
from langchain_community.document_loaders.parsers.pdf import PyPDFium2Parser
from langchain_community.vectorstores import FAISS  # pip install faiss-cpu
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List, Dict, Any, Optional, Callable
import uuid
import tiktoken  # pip install tiktoken
import pickle
from pathlib import Path
import shutil
import json


# FastAPI related imports
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse
import uvicorn
from contextlib import asynccontextmanager
import redis
from pydantic import BaseModel
import numpy as np
from config import API_KEY

# ===================== VectorDataBase Class (No Modifications) =====================
class VectorDataBase(FAISS):

    def __init__(
        self,
        embedding_function,
        index,
        docstore,
        index_to_docstore_id,
        relevance_score_fn: Optional[Callable[[float], float]] = None,
        normalize_L2: bool = False,
        distance_strategy: DistanceStrategy = DistanceStrategy.EUCLIDEAN_DISTANCE,
        id: Optional[str] = None
    ):
        super().__init__(
            embedding_function=embedding_function,
            index=index,
            docstore=docstore,
            index_to_docstore_id=index_to_docstore_id,
            relevance_score_fn=relevance_score_fn,
            normalize_L2=normalize_L2,
            distance_strategy=distance_strategy
        )
        self.id = id or str(uuid.uuid4())

    def save_local(self, folder_path: str, index_name: str = "index") -> None:
        super().save_local(folder_path, index_name)
        path = Path(folder_path)
        path.mkdir(exist_ok=True, parents=True)
        with open(path / f"{index_name}_id.pkl", "wb") as f:
            pickle.dump({"id": self.id}, f)

    @classmethod
    def load_local(
        cls,
        folder_path: str,
        embeddings,
        index_name: str = "index",
        *,
        allow_dangerous_deserialization: bool = False,
        **kwargs
    ) -> "VectorDataBase":

        base_instance = super().load_local(
            folder_path=folder_path,
            embeddings=embeddings,
            index_name=index_name,
            allow_dangerous_deserialization=allow_dangerous_deserialization, 
            **kwargs
        )

        path = Path(folder_path)
        id = None
        try:
            with open(path / f"{index_name}_id.pkl", "rb") as f:
                id_data = pickle.load(f)
                id = id_data.get("id")
        except FileNotFoundError:
            id = str(uuid.uuid4())

        instance = cls(
            embedding_function=base_instance.embedding_function,
            index=base_instance.index,
            docstore=base_instance.docstore,
            index_to_docstore_id=base_instance.index_to_docstore_id,
            relevance_score_fn=base_instance.override_relevance_score_fn,
            normalize_L2=base_instance._normalize_L2,
            distance_strategy=base_instance.distance_strategy,
            id=id
        )
        return instance

    @classmethod
    def from_documents(
        cls,
        documents: list[Document],
        embedding,
        id: Optional[str] = None,** kwargs: Any,
    ) -> "VectorDataBase":

        base_faiss = super().from_documents(documents, embedding, **kwargs)
        vector_db = cls(
            embedding_function=base_faiss.embedding_function,
            index=base_faiss.index,
            docstore=base_faiss.docstore,
            index_to_docstore_id=base_faiss.index_to_docstore_id,
            relevance_score_fn=base_faiss.override_relevance_score_fn,
            normalize_L2=base_faiss._normalize_L2,
            distance_strategy=base_faiss.distance_strategy,
            id=id
        )
        return vector_db

# ===================== RAGTool Class (Fixed Version) =====================
class RAGTool:

    def __init__(
        self,
        embedding_api_key: str,
        embedding_model: str = "text-embedding-v3",
        chunk_size: int = 500,
        chunk_overlap: int = 200,
        extract_image: bool = False  # 新增extract_image属性
    ):
        self.embedding_api_key = embedding_api_key
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size 
        self.chunk_overlap = chunk_overlap 
        self.extract_image = extract_image  # 初始化提取图片配置

        self.embedding = DashScopeEmbeddings(
            dashscope_api_key=self.embedding_api_key,
            model=self.embedding_model
        )
        self.pdf_chunks: List[Dict] = []
        self.vector_db: Optional[VectorDataBase] = None
        self.file_info: Dict[str, Any] = {}
        self.token_encoder = tiktoken.get_encoding("cl100k_base")
        self.vectorbase_usage = 0
        self.query_usage = 0

    def count_tokens(self, text: str) -> int:
        return len(self.token_encoder.encode(text)) if text else 0

    def process_pdf(
        self, 
        file_content: bytes, 
        filename: str, 
        extract_images: Optional[bool] = None  
    ) -> List[Dict]:
        try:
            # 优先使用传入的extract_images，否则用初始化的extract_image
            extract_img = extract_images if extract_images is not None else self.extract_image
            
            blob = Blob.from_data(
                data=file_content,
                path=filename
            )

            parser = PyPDFium2Parser(
                extract_images=extract_img,  # 使用最终的提取配置
                images_inner_format="html-img"
            )
            documents = list(parser.parse(blob))

            full_content = "\n\n".join([doc.page_content for doc in documents])

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=["\n\n", "\n", " ", ""]
            )
            all_chunks = splitter.split_text(full_content)

            chunks = []
            self.vectorbase_usage = 0
            total_chunks = len(all_chunks)
            for i, chunk_text in enumerate(all_chunks):
                clean_text = chunk_text.strip()
                if not clean_text:
                    continue
                chunk_token = self.count_tokens(clean_text)
                self.vectorbase_usage += chunk_token
                chunks.append({
                    "id": str(uuid.uuid4()),
                    "content": clean_text,
                    "metadata": {
                        "filename": filename,
                        "chunk_index": i,
                        "total_chunks": total_chunks,
                        "token": chunk_token
                    }
                })

            self.pdf_chunks = chunks
            self.file_info = {
                "filename": filename,
                "chunk_count": len(chunks),
                "file_size": len(file_content),
                "extract_image": extract_img,  # 记录实际使用的提取配置
                "vectorbase_usage": self.vectorbase_usage
            }

            return chunks

        except Exception as e:
            raise RuntimeError(f"Failed to chunk PDF: {str(e)}")

    def build_vector_db(self, vector_db_id: Optional[str] = None) -> VectorDataBase:
        if not self.pdf_chunks:
            raise RuntimeError("Please process PDF with process_pdf first before building vector database")

        try:
            langchain_docs = []
            for chunk in self.pdf_chunks:
                langchain_doc = Document(
                    page_content=chunk["content"],
                    metadata={
                        "id": chunk["id"],
                        "filename": chunk["metadata"]["filename"],
                        "chunk_index": chunk["metadata"]["chunk_index"],
                        "total_chunks": chunk["metadata"]["total_chunks"]
                    }
                )
                langchain_docs.append(langchain_doc)

            self.vector_db = VectorDataBase.from_documents(
                documents=langchain_docs,
                embedding=self.embedding,
                id=vector_db_id
            )
            self.file_info["vector_db_id"] = self.vector_db.id

            return self.vector_db

        except Exception as e:
            raise RuntimeError(f"Failed to build FAISS vector database: {str(e)}")

    def semantic_search(self, query: str, top_k: int = 3) -> List[Dict]:
        if not self.vector_db:
            raise RuntimeError("Please build vector database first before performing search")

        try:
            query_token = self.count_tokens(query)
            self.query_usage += query_token

            results_with_score = self.vector_db.similarity_search_with_score(query, k=top_k)

            formatted_results = []
            for doc, score in results_with_score:
                similarity_score = float(score) if isinstance(score, np.float32) else round(score, 4)
                formatted_results.append({
                    "id": doc.metadata["id"],
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "similarity_score": similarity_score,
                    "vector_db_id": self.vector_db.id
                })

            return formatted_results

        except Exception as e:
            raise RuntimeError(f"Semantic search failed: {str(e)}")

    def save_vector_db(self, folder_path: str, index_name: str = "index") -> None:
        if not self.vector_db:
            raise RuntimeError("Please build vector database first before saving")
        self.vector_db.save_local(folder_path, index_name)
        print(f"Vector database (id: {self.vector_db.id}) saved to: {folder_path}")

    def load_vector_db(self, id: str) -> VectorDataBase:
        try:
            folder_path = f"./{id}"
            index_name = id
            self.vector_db = VectorDataBase.load_local(
                folder_path=folder_path,
                index_name=index_name,
                embeddings=self.embedding,
                allow_dangerous_deserialization=True
            )
            if self.vector_db.id != id:
                raise RuntimeError(f"Unmatched: {self.vector_db.id} != {id}")
            self.file_info["vector_db_id"] = id
            return self.vector_db
        except FileNotFoundError:
            raise RuntimeError(f"VectorBase-{id} not found in {folder_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load vector database: {str(e)}")

    def get_token_stats(self) -> Dict:
        return {
            "vectorbase_usage": self.vectorbase_usage,
            "query_usage": self.query_usage,
            "total_usage": self.vectorbase_usage + self.query_usage
        }

    def get_file_info(self) -> Dict:
        return self.file_info

    def clear(self):
        self.pdf_chunks = []
        self.vector_db = None
        self.file_info = {}
        self.vectorbase_usage = 0
        self.query_usage = 0

    def clear_disk(self, by_id: bool = False, vector_db_id: Optional[str] = None) -> None:
        try:
            if by_id:
                if not vector_db_id:
                    raise RuntimeError("vector_db_id is required when by_id=True")
                target_path = Path(f"./{vector_db_id}")
                if not target_path.exists():
                    raise RuntimeError(f"Vector database {vector_db_id} not found on disk")
                if target_path.is_dir():
                    shutil.rmtree(target_path)
                    print(f"Successfully deleted vector database {vector_db_id} from {target_path}")
                elif target_path.is_file():
                    target_path.unlink()
                    print(f"Successfully deleted vector database file {vector_db_id}")
            else:
                deleted_count = 0
                current_dir = Path(".")
                for item in current_dir.iterdir():
                    if item.is_dir():
                        try:
                            uuid.UUID(item.name)
                            shutil.rmtree(item)
                            deleted_count += 1
                            print(f"Deleted vector database: {item}")
                        except ValueError:
                            continue
                if deleted_count == 0:
                    print("No vector databases found on disk to delete")
                else:
                    print(f"Successfully deleted {deleted_count} vector databases")
        except PermissionError:
            raise RuntimeError("Permission denied: cannot delete vector database files")
        except Exception as e:
            raise RuntimeError(f"Failed to clear vector database from disk: {str(e)}")


# Initialize Redis client (modify according to your Redis configuration)
redis_client = redis.Redis(
    host="localhost",    # Redis address
    port=6379,           # Redis port
    db=0,                # Database number to use
    password="milo_2357",         # Redis password (leave empty if none)
    decode_responses=True  # Auto decode to string (avoid bytes type)
)

# Custom serialization/deserialization tools (store key info of RAGTool, not the entire instance)
def serialize_rag_info(rag_tool: "RAGTool") -> str:
    info = {
        "embedding_api_key": rag_tool.embedding_api_key,
        "embedding_model": rag_tool.embedding_model,
        "chunk_size": rag_tool.chunk_size,
        "chunk_overlap": rag_tool.chunk_overlap,
        "extract_image": rag_tool.extract_image,  # 现在该属性已存在
        "file_info": rag_tool.file_info,
        "token_stats": rag_tool.get_token_stats()
    }
    return json.dumps(info)

def deserialize_rag_info(info_str: str) -> Dict:
    return json.loads(info_str)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        kb_ids = redis_client.keys("rag:kb:*")
        for kb_key in kb_ids:
            kb_id = kb_key.replace("rag:kb:", "")
            if not Path(f"./{kb_id}").exists():
                redis_client.delete(kb_key)
    except:
        pass
    yield
    pass


app = FastAPI(
    title="PDF AgenticRAG API",
    description="Upload PDF to build vector database, support semantic retrieval by knowledge base ID (Redis Version)",
    version="1.0",
    lifespan=lifespan
)

class SearchRequest(BaseModel):
    knowledge_base_id: str
    query: str
    top_k: int = 3  

@app.get("/")
async def root(): 
    return {"message": "RAG API Server is running..."}

@app.get("/health", summary="Health check", response_description="Return server status")
async def health():
    redis_status = "ok" if redis_client.ping() else "failed"
    kb_count = len(redis_client.keys("rag:kb:*"))
    return {
        "status": "ok",
        "redis_status": redis_status,
        "active_knowledge_bases": kb_count,
        "knowledge_base_ids": [key.replace("rag:kb:", "") for key in redis_client.keys("rag:kb:*")]
    }

# 上传文件并构建知识库
@app.post("/upload-pdf")
async def upload_pdf(
    file: UploadFile = File(..., description="PDF file to upload"),
    extract_images: bool = Query(False, description="Whether to extract images from PDF")
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    try:
        file_content = await file.read()
        if len(file_content) == 0:
            raise HTTPException(status_code=400, detail="Empty PDF file")
        
        rag_tool = RAGTool(
            embedding_api_key=API_KEY,
            chunk_size=500,       # 块大小
            chunk_overlap=200,    # 重叠长度
        )

        # 调用process_pdf，传入extract_images参数
        rag_tool.process_pdf(
            file_content=file_content,
            filename=file.filename,
            extract_images=extract_images
        )
        
        vector_db = rag_tool.build_vector_db()
        kb_id = vector_db.id
        rag_tool.save_vector_db(folder_path=f"./{kb_id}", index_name=kb_id)
        redis_key = f"rag:kb:{kb_id}"
        redis_client.set(
            name=redis_key,
            value=serialize_rag_info(rag_tool),
            ex=86400  # 24 hours: to avoid Redis data redundancy
        )
        # 存储文件名到kb_id的映射
        filename_key = f"rag:filename:{file.filename}"
        redis_client.set(
            name=filename_key,
            value=kb_id,
            ex=86400  # 24 hours: 与知识库信息保持一致
        )
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "message": "PDF uploaded and vector database built successfully",
                "data": {
                    "knowledge_base_id": kb_id,
                    "filename": file.filename,
                    "file_size": len(file_content),
                    "chunk_count": len(rag_tool.pdf_chunks),
                    "token_stats": rag_tool.get_token_stats()  # 新增返回Token统计
                }
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {str(e)}")

@app.post("/search", summary="Semantic retrieval (POST version)")
async def semantic_search_post(request: SearchRequest):
    """Semantic retrieval by knowledge base ID (POST version)"""
    try:
        # 提取请求参数
        kb_id = request.knowledge_base_id
        query = request.query
        top_k = request.top_k

        # 基础校验 & 加载知识库
        redis_key = f"rag:kb:{kb_id}"
        kb_folder = Path(f"./{kb_id}")

        if not redis_client.exists(redis_key) and not kb_folder.exists():
            raise HTTPException(
                status_code=404, 
                detail=f"Knowledge base {kb_id} not found"
            )
        rag_tool = RAGTool(embedding_api_key=API_KEY)
        
        # 从redis中拿到缓存信息
        if redis_client.exists(redis_key):
            rag_info = deserialize_rag_info(redis_client.get(redis_key))
            rag_tool.vectorbase_usage = rag_info["token_stats"]["vectorbase_usage"]
            rag_tool.chunk_size = rag_info["chunk_size"]
            rag_tool.chunk_overlap = rag_info["chunk_overlap"]
            rag_tool.extract_image = rag_info["extract_image"]
            rag_tool.file_info = rag_info["file_info"]
        
        # 加载向量库
        try:
            rag_tool.load_vector_db(id=kb_id)
            if not redis_client.exists(redis_key):
                redis_client.set(redis_key, serialize_rag_info(rag_tool), ex=86400)
        except Exception as e:
            raise HTTPException(
                status_code=404, 
                detail=f"Knowledge base corrupted: {str(e)}"
            )

        
        # 执行检索（已在semantic_search中处理float32）
        results = rag_tool.semantic_search(query=query, top_k=top_k)

        # 返回响应（vectorbase_usage已恢复）
        return {
            "success": True,
            "message": "Search success",
            "data": {
                "knowledge_base_id": kb_id,
                "query": query,
                "top_k": top_k,
                "results": results,
                "token_stats": rag_tool.get_token_stats()
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.get("/get-kb-id-by-filename")
async def get_kb_id_by_filename(filename: str = Query(..., description="Filename to lookup")):
    try:
        filename_key = f"rag:filename:{filename}"
        kb_id = redis_client.get(filename_key)
        
        # 如果Redis中没有，尝试从所有知识库中查找
        if not kb_id:
            kb_ids = redis_client.keys("rag:kb:*")
            for kb_key in kb_ids:
                kb_info_str = redis_client.get(kb_key)
                if kb_info_str:
                    kb_info = deserialize_rag_info(kb_info_str)
                    if kb_info.get("file_info", {}).get("filename") == filename:
                        kb_id = kb_key.replace("rag:kb:", "")
                        redis_client.set(filename_key, kb_id, ex=86400)
                        break
            
            if not kb_id:
                raise HTTPException(
                    status_code=404,
                    detail=f"Knowledge base not found for filename: {filename}"
                )
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "data": {
                    "filename": filename,
                    "knowledge_base_id": kb_id
                }
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get kb_id: {str(e)}")

@app.delete("/delete-knowledge-base")
async def delete_knowledge_base(knowledge_base_id: str = Query(..., description="Knowledge base ID")):
    redis_key = f"rag:kb:{knowledge_base_id}"
    if not redis_client.exists(redis_key) and not Path(f"./{knowledge_base_id}").exists():
        raise HTTPException(status_code=404, detail=f"Knowledge base {knowledge_base_id} not found")

    try:
        # 获取文件名信息，删除文件名映射
        if redis_client.exists(redis_key):
            kb_info_str = redis_client.get(redis_key)
            if kb_info_str:
                kb_info = deserialize_rag_info(kb_info_str)
                filename = kb_info.get("file_info", {}).get("filename")
                if filename:
                    filename_key = f"rag:filename:{filename}"
                    redis_client.delete(filename_key)
        
        redis_client.delete(redis_key)
        rag_tool = RAGTool(embedding_api_key=API_KEY)
        rag_tool.clear_disk(by_id=True, vector_db_id=knowledge_base_id)

        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "message": f"Knowledge base {knowledge_base_id} deleted successfully"
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Delete failed: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=5000,
        log_level="info"
    )