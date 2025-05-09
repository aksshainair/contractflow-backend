from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Any, AsyncGenerator
import os
import logging
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter
from openai import OpenAI
from qdrant_client.models import Filter, FieldCondition, MatchValue
from .embedding_utils import embed_text
import json
import asyncio
from .database import (
    get_user_by_email, create_user, create_document,
    get_document_by_id, update_document, get_documents_for_user,
    get_user_by_id
)

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[logging.StreamHandler()]
)

# Load environment variables
load_dotenv()

# Get environment variables
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "contract-openai"
EMBEDDING_MODEL = "text-embedding-3-small"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
REASONING_MODEL = "gpt-4o"

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

router = APIRouter()

class ChatQuery(BaseModel):
    query: str
    document_id: str
    filetype: str = "contract"
    top_k: int = 3

class ChatResponse(BaseModel):
    response: str

async def stream_llm_response(query: str, search_results: List[Dict]):
    """Stream the LLM response."""
    try:
        # Format the search results
        formatted_results = []
        for result in search_results:
            formatted_results.append({
                'id': result['id'],
                'score': result['score'],
                'payload': result['payload']
            })

        # Create the system and user prompts
        system_prompt = """
    You are a helpful, professional legal document analyzer assistant. You will be provided with some data in a json array format from a vector database and a user query.
    Your task is to analyze the data and provide a detailed and structured response.
    The data can be from an invoice, contract, purchase order, or any other legal document.
    Use proper formating and structure in your response.
    If you have any conflict while descision making, don't hesitate to ask for clarification and tell what conflict or problem you are facing.
    While answering questions including tables, you must look at the column names and data types of the table, also note that some cells can be empty.
    You don't have to include the source of the data in your response. 
    Your response shouldn't have to cite them or give any indication of the backend data or database.
    As this is a client facing application, you should not include any internal information or any information about the database.
    You should not include any information about the database or the data source.
    """

        user_prompt = f"""
    Here is the data from the database:
    {formatted_results}
    Here is the query:
    {query}
    """

        # Invoke the LLM with streaming
        print("Invoking the LLM with system and user prompts for streaming.")
        stream = client.chat.completions.create(
            model=REASONING_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            stream=True
        )

        # Stream the response
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                yield f"data: {chunk.choices[0].delta.content}\n\n"

    except Exception as e:
        print(f"Error in streaming LLM response: {str(e)}")
        import traceback
        traceback.print_exc()
        yield f"data: Error: {str(e)}\n\n"

def llm_query(data: List[Dict[str, Any]], query: str, filetype: str) -> str:
    """
    Function to query the LLM with the provided data from the qdrant db results.
    """
    system_prompt = f"""
    You are a helpful, professional legal document analyzer assistant. You will be provided with some data in a json array format from a vector database and a user query.
    Your task is to analyze the data and provide a detailed and structured response.
    The data can be from an invoice, contract, purchase order, or any other legal document.
    Use proper formating and structure in your response.
    If you have any conflict while descision making, don't hesitate to ask for clarification and tell what conflict or problem you are facing.
    While answering questions including tables, you must look at the column names and data types of the table, also note that some cells can be empty.
    You should not include the source of the data in your response. 
    Your response shouldn't have to cite them or give any indication of the backend data or database.
    As this is a client facing application, you should not include any internal information or any information about the database.
    You should not include any information about the database or the data source.
    The response must be structured and not in markdown. 
    """

    user_prompt = f"""
    Here is the data from the database:
    {data}
    Here is the query:
    {query}
    """

    try:
        logging.debug("Invoking the LLM with system and user prompts.")
        response = client.chat.completions.create(
            model=REASONING_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
        )
        logging.info("LLM query executed successfully.")
        return response.choices[0].message.content
    except Exception as e:
        logging.error(f"Error invoking LLM: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

def generalized_query(query: str, document_id: str, filetype: str, top_k: int = 3) -> Dict[str, Any]:
    """
    Perform a generalized search in the Qdrant collection and return the top results.
    """
    try:
        logging.debug("Initializing Qdrant client.")
        qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

        # Get the document from the database to get its filename
        document = get_document_by_id(document_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")

        filename = document['title']
        filename = filename.replace(".sfdt", ".docx")
        print(f"filename  : {filename}")
        logging.debug(f"Applying filter on filename: {filename}")
        search_filter = Filter(
            must=[
                FieldCondition(
                    key="filename",
                    match=MatchValue(value=filename)
                )
            ]
        )

        logging.debug("Generating embedding for the query.")
        query_embedding = embed_text(query)

        logging.debug("Performing search in the Qdrant collection.")
        results = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_embedding,
            limit=top_k,
            with_payload=True,
            query_filter=search_filter
        )

        logging.debug("Formatting the search results.")
        formatted_results = []
        for result in results:
            logging.debug(f"Processing result with ID: {result.id}, Score: {result.score}")
            formatted_results.append({
                "id": result.id,
                "score": result.score,
                "payload": result.payload,
            })

        logging.info("Query processed successfully.")
        return {
            "query": query,
            "results": formatted_results,
        }

    except Exception as e:
        logging.error(f"Error processing query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(chat_query: ChatQuery):
    """
    Endpoint to handle AI chat queries.
    """
    try:
        # Run the vector query
        vector_results = generalized_query(
            query=chat_query.query,
            document_id=chat_query.document_id,
            filetype=chat_query.filetype,
            top_k=chat_query.top_k
        )

        # Run the LLM query
        llm_response = llm_query(
            data=vector_results['results'],
            query=chat_query.query,
            filetype=chat_query.filetype
        )

        return ChatResponse(response=llm_response)

    except Exception as e:
        logging.error(f"Error in chat endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/chat/stream")
async def chat_stream_endpoint(chat_query: ChatQuery):
    """
    Streaming endpoint to handle AI chat queries with real-time responses.
    """
    try:
        # Run the vector query
        vector_results = generalized_query(
            query=chat_query.query,
            document_id=chat_query.document_id,
            filetype=chat_query.filetype,
            top_k=chat_query.top_k
        )

        # Stream the LLM response
        return StreamingResponse(
            stream_llm_response(
                query=chat_query.query,
                search_results=vector_results['results']
            ),
            media_type="text/event-stream"
        )

    except Exception as e:
        logging.error(f"Error in chat stream endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) 