"""RMB Backend ChatBot Application"""

import pysqlite3 as sqlite3
import sys
import os
sys.modules['sqlite3'] = sqlite3
# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import asyncio
import logging
import json
import time
import re
import dotenv
import aiohttp
from aiohttp import ClientSession, TCPConnector
import uvicorn
from openai import AzureOpenAI
from fastapi import FastAPI, HTTPException, Request, Form, File, UploadFile, Depends, Security
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security.api_key import APIKeyHeader, APIKey
from fastapi.middleware.cors import CORSMiddleware
from weaviate import Client as WeaviateClient
from weaviate.auth import AuthApiKey
from app.common.common_code import (
    format_options,
    with_internet_context,
    get_and_trim_entities
)
from app.api.internet.internet_service import internet_search_async
from app.api.file.weaviate_service import (
    add_pages_to_document,
    delete_weaviate_object,
    extract_pages_tables,
    get_weaviate_client,
    recursive_chunks,
    retrieve_session_data,
    process_file,
    create_user_details,
    delete_userinfo_by_user_id
)

# Import query rewriter prompts
from app.query_rewriter_prompts.default_prompt import default_prompt
from app.query_rewriter_prompts.executive_summary_prompt import executive_summary_prompt
from app.query_rewriter_prompts.key_risks_prompt import key_risks_prompt
from app.query_rewriter_prompts.market_overview_prompt import market_overview_prompt


# Load environment variables
dotenv.load_dotenv()

# Read all the config variables from .env file into global variables
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
OPENAI_API_VERSION = os.getenv("OPENAI_API_VERSION")
DEPLOYMENT_NAME = os.getenv("DEPLOYMENT_NAME")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
# Document Intelligence Credentials
DOCINTELLIGENCE_ENDPOINT= os.getenv("DOCINTELLIGENCE_ENDPOINT")
DOCINTELLIGENCE_KEY= os.getenv("DOCINTELLIGENCE_KEY")
# Weaviate Credentials
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")
WEAVIATE_URL = os.getenv("WEAVIATE_URL")
# Bing Search Credentials
BING_SEARCH_API_KEY = os.getenv("BING_SEARCH_API_KEY")
BING_SEARCH_ENDPOINT = os.getenv("BING_SEARCH_ENDPOINT")
# Authentication credentials
API_KEY = os.getenv("API_KEY")
API_KEY_NAME = os.getenv("API_KEY_NAME")
# TextAnalytics Credentials
TEXT_ANALYTICS_AZURE_KEY = os.getenv("TEXT_ANALYTICS_AZURE_KEY")
TEXT_ANALYTICS_AZURE_ENDPOPINT = os.getenv("TEXT_ANALYTICS_AZURE_ENDPOPINT")
# NERtags
EXTRACT_NERTAGS = os.getenv("EXTRACT_NERTAGS")

# Initialize Weaviate client
weaviate_client = WeaviateClient(
    url=WEAVIATE_URL,
    auth_client_secret=AuthApiKey(api_key=WEAVIATE_API_KEY)
)

# Alternative Questions
num_alternatives = os.getenv("NUM_ALTERNATIVES")

header_api_key = APIKeyHeader(name=API_KEY_NAME, auto_error=False)


def get_api_key(api_key_header: str = Security(header_api_key)):
    if api_key_header == API_KEY:
        return api_key_header
    raise HTTPException(status_code=400, detail="Invalid API Key")


# Initialize Azure OpenAI client
embedding_client = AzureOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    api_version=OPENAI_API_VERSION,
)

# Initialize FastAPI
app = FastAPI()

# Initialize CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set the logging level to DEBUG for detailed logs
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler()  # This will log to standard output
    ]
)
logger = logging.getLogger(__name__)

# Streaming generator function
async def azure_stream(messages):
    async with aiohttp.ClientSession() as session:
        try:
            url = f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/{DEPLOYMENT_NAME}/chat/completions?api-version={OPENAI_API_VERSION}"
            headers = {
                "api-key": AZURE_OPENAI_API_KEY,
                "Content-Type": "application/json",
            }
            payload = {"messages": messages, "stream": True}
            logging.debug(
                "Sending request to %s with headers %s and payload %s",
                url,
                headers,
                payload,
            )
            async with session.post(url, headers=headers, json=payload) as response:
                if response.status != 200:
                    error_detail = await response.text()
                    logging.error(
                        "Request failed with status %s: %s",
                        response.status,
                        error_detail,
                    )
                    raise HTTPException(
                        status_code=response.status, detail=error_detail
                    )

                async for line in response.content:
                    if line:
                        line_data = line.decode("utf-8").strip()
                        if line_data == "data: [DONE]":
                            break
                        if line_data.startswith("data:"):
                            json_data = line_data[5:]
                            try:
                                data_dict = json.loads(json_data)
                                choices = data_dict.get("choices", [])
                                for choice in choices:
                                    if (
                                        "delta" in choice
                                        and "content" in choice["delta"]
                                    ):
                                        yield choice["delta"]["content"]
                            except json.JSONDecodeError:
                                logging.error("Error decoding JSON: %s", json_data)
        except aiohttp.ClientError as e:
            logging.error("Client error: %s", str(e))
            raise HTTPException(status_code=500, detail=str(e)) from e

def embed_and_query_weaviate_hybrid(query, embedding_model, user_ID) -> dict:
    """
    Embeds the query using the specified embedding model and performs a hybrid query in Weaviate.
    
        dict: A dictionary containing the filtered results of the hybrid query.
        query (str): The input query.
        embedding_model (str): The model for creating the query embedding.

    Returns:
        dict: The search results from Weaviate.

    Raises:
        Exception: If an error occurs during embedding or querying.
    """
    try:
        # Create an embedding for the query
        vector = embedding_client.embeddings.create(
            model=embedding_model,
            input=query
        )
        query_vector = vector.data[0].embedding

        # Define the filter clause
        filter_clause = {
            "operator": "Equal",
            "path": ["document", "Document", "user_info", "UserInfo", "user_ID"],
            "valueString": user_ID
        }

        result = weaviate_client.query.get(
            "DocumentPages",
            [
                "content",
                "num_tokens",

                "_additional { id }",
                "document { ... on Document { title web_url user_info { ... on UserInfo { user_ID } } client { ... on Clients { company_name } } document_type { ... on DocumentTypes { document_type } } } }"

            ]
        ).with_hybrid(
            query=query,
            vector=query_vector
        ).with_where(
            filter_clause
        ).with_limit(3).do()

        # Log the result for debugging
        logging.info(f"Query result: {result}")

        if not result.get('data', {}).get('Get', {}).get('DocumentPages'):
            logging.error("Unexpected response structure or no results found.")
            return None

        # Filter results to ensure only correct user_ID
        filtered_results = []
        for document_page in result['data']['Get']['DocumentPages']:
            documents = document_page.get('document', [])
            if isinstance(documents, list):
                for document in documents:
                    user_info = document.get('user_info', [])
                    if isinstance(user_info, list) and any(user['user_ID'] == user_ID for user in user_info):
                        filtered_results.append(document_page)

        logging.info(f"Filtered results: {filtered_results}")
        return {'data': {'Get': {'DocumentPages': filtered_results}}}
    except Exception as ex:
        logging.error(f"Error in querying Weaviate: {ex}")
        raise

async def generate_alternative_questions(question: str, num_alternatives: int, format_option: str = "DEFAULT_PROMT"):
    """
    Generates alternative questions based on the given question and format option.

    Args:
        question (str): The input question.
        num_alternatives (int): The number of alternative questions to generate.
        format_option (str, optional): The format option for the prompt. Defaults to "DEFAULT_PROMT".

    Returns:
        list: A list of generated alternative questions.

    Raises:
        HTTPException: If the request to the OpenAI API fails.
    """
    logging.info(f"Received format_option: '{format_option}', type: {type(format_option)}")
    logging.info("Generating alternative questions for the query: '%s'", question)

    url = f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/{DEPLOYMENT_NAME}/chat/completions?api-version={OPENAI_API_VERSION}"
    headers = {"api-key": AZURE_OPENAI_API_KEY, "Content-Type": "application/json"}

    if format_option == "EXECUTIVE_SUMMARY_PROMPT":
        prompt = executive_summary_prompt
        logging.info("Using EXECUTIVE_SUMMARY_PROMPT")
    elif format_option == "KEY_RISKS_PROMPT":
        prompt = key_risks_prompt
        logging.info("Using KEY_RISKS_PROMPT")
    elif format_option == "MARKET_OVERVIEW_PROMPT":
        prompt = market_overview_prompt
        logging.info("Using MARKET_OVERVIEW_PROMPT")
    else:
        logging.info(f"Using DEFAULT_PROMT, format_option: '{format_option}'")
        prompt = default_prompt

    # Format the prompt with the question and number of alternatives
    prompt = prompt.format(question=question, num_alternatives=num_alternatives)

    payload = {
        "messages": [{"role": "user", "content": prompt}]
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=payload) as response:
            if response.status != 200:
                error_detail = await response.text()
                logging.error("Request failed with status %s: %s", response.status, error_detail)
                raise HTTPException(status_code=response.status, detail=error_detail)

            response_data = await response.json()
            output = response_data["choices"][0]["message"]["content"]
            questions = re.findall(r"<alternative_questions>\s*(.*?)\s*</alternative_questions>", output, re.DOTALL)
            if questions:
                question_list = questions[0].split("\n")
                question_list = [q.strip() for q in question_list if q.strip()]
                logging.info("Generated %d alternative questions: %s", len(question_list), ", ".join(question_list))
                return question_list
            logging.warning("No alternative questions were generated")
            return []


@app.post("/weaviate-search")
async def weaviate_search(request: Request, api_key: APIKey = Depends(get_api_key)):
    """
    Handles a POST request to perform a Weaviate search with alternative question generation.
    Args:
        request (Request): The request object containing search parameters.
        api_key (APIKey): The API key for authentication, obtained through dependency injection.
    Returns:
        StreamingResponse: A streaming response with the search results.
    Raises:
        HTTPException: If the query parameter is missing.

        JSONResponse: If an error occurs during processing.
    """

    try:
        data = await request.json()
        prompt = data.get("query", "")
        format_option = data.get("format_option", "")
        messages = data.get("messages", [])
        internet_rag = data.get("internet_rag", False)
        session_id = data.get("session_id", "")

        user_id = data.get("user_id", "")
        num_alternative = data.get("num_alternatives", 2)

        if not prompt:
            raise HTTPException(status_code=400, detail="query is required.")

        alternative_questions = await generate_alternative_questions(prompt, num_alternative, format_option)
        if not alternative_questions:
            logging.warning("No alternative questions were generated. Using original query.")
            alternative_questions = [prompt]

        logging.info("Questions to be processed: %s", alternative_questions)
        unique_chunks = {}
        weaviate_contexts = []
        internet_contexts = []
        output_links = []
        all_successful_urls = []
        seen_urls = set()
        seen_titles = set()

        connector = TCPConnector(ssl=False)
        async with ClientSession(connector=connector) as session:
            for alt_query in alternative_questions:
                weaviate_context = embed_and_query_weaviate_hybrid(alt_query, EMBEDDING_MODEL, user_id)

                if isinstance(weaviate_context, dict) and 'data' in weaviate_context:
                    if 'Get' in weaviate_context['data'] and 'DocumentPages' in weaviate_context['data']['Get']:
                        documents = weaviate_context['data']['Get']['DocumentPages']
                        logging.info(f"Documents retrieved: {documents}")

                        for doc in documents:
                            if isinstance(doc, dict) and 'content' in doc:
                                chunk_id = doc.get('_additional', {}).get('id', str(hash(doc['content'])))
                                if chunk_id not in unique_chunks:
                                    unique_chunks[chunk_id] = doc
                                    if 'document' in doc and isinstance(doc['document'], list) and len(doc['document']) > 0:
                                        document_info = doc['document'][0]
                                        web_url = document_info.get('web_url', '')
                                        title = document_info.get('title', '')
                                        if web_url and title and web_url not in seen_urls and title not in seen_titles:
                                            output_links.append(f"[{title}]({web_url})\n")
                                            seen_urls.add(web_url)
                                            seen_titles.add(title)
                                    weaviate_contexts.append(doc['content'])
                    else:
                        logging.warning(f"Unexpected Weaviate response structure for query '{alt_query}'")
                else:
                    logging.warning(f"Unexpected Weaviate response type for query '{alt_query}'")

                if internet_rag:
                    internet_context, successful_urls = await internet_search_async(alt_query, session_id, session)
                    all_successful_urls.extend(successful_urls)
                    internet_contexts.append(internet_context)

        combined_weaviate_context = "\n\n".join(weaviate_contexts)
        combined_internet_context = "\n\n".join(internet_contexts) if internet_rag else ""

        context = (
            with_internet_context(combined_weaviate_context, combined_internet_context)
            if internet_rag
            else combined_weaviate_context
        )

        prompt_messages = message_prompt(prompt, context, format_option, messages)

        # Return both the streaming response and the weaviate_context data
        response_data = {
            "weaviate_links": output_links,
            "internet_links": all_successful_urls,
        }

        response_headers = {
            "response_data": json.dumps(response_data)
        }

        # Return the StreamingResponse along with response data
        return StreamingResponse(azure_stream(prompt_messages), media_type="text/event-stream", headers=response_headers)

    except Exception as error:
        logging.error(f"Error in /weaviate-search: {str(error)}")
        return JSONResponse(content={"error": str(error)}, status_code=500)

@app.post("/internet-search")
async def internet_search(request: Request, api_key: APIKey = Depends(get_api_key)):
    """
    Handle a POST request to perform an internet search based on the provided query.

    Args:
        request (Request): The incoming HTTP request containing the search query and options.
        api_key (APIKey, optional): API key for authentication.

    Returns:
        StreamingResponse: A streaming response containing the search results.

    Raises:
        HTTPException: If the 'query' or 'session_id' fields are missing in the request body.
        JSONResponse: If an error occurs during processing.
    """
    start_time = time.time()
    try:
        data = await request.json()
        original_query = data.get("query", "")
        session_id = data.get("session_id", "")
        format_option = data.get("format_option", "")
        messages = data.get("messages", [])
        num_alternative = data.get("num_alternatives", num_alternatives)

        if not original_query or not session_id:
            raise HTTPException(
                status_code=400, detail="query and session_id are required."
            )

        logging.info("Starting internet search for query %s:",original_query)

        # Generate alternative questions
        alt_questions_start = time.time()
        alternative_questions = await generate_alternative_questions(
            original_query, num_alternative, format_option
        )
        alt_questions_end = time.time()

        # If no alternative questions were generated, use the original query
        if not alternative_questions:
            logging.warning("No alternative questions were generated. Using original query.")
            alternative_questions = [original_query]

        logging.info(
            "Generated %d questions in %.2f seconds",
            len(alternative_questions),
            alt_questions_end - alt_questions_start
        )

        logging.info("Questions to be processed: %s", alternative_questions)

        # Create an aiohttp ClientSession with SSL verification disabled
        connector = TCPConnector(ssl=False)
        async with ClientSession(connector=connector) as session:
            # Run internet searches for all questions in parallel
            search_start = time.time()
            results = await asyncio.gather(
                *[
                    internet_search_async(query, session_id, session)
                    for query in alternative_questions
                ]
            )
            search_end = time.time()
            logging.info("Completed internet searches in %.2f seconds", search_end - search_start)

        # Separate contexts and successful URLs
        contexts, all_successful_urls = zip(*results)

        # Separate contexts and successful URLs
        contexts, all_successful_urls = zip(*results)
        
        # Combine the contexts
        combined_context = "\n\n".join(contexts)
        
        # Flatten the list of successful URLs
        successful_urls = [url for sublist in all_successful_urls for url in sublist]
        
        # Log all successful URLs
        logging.info(f"All successful URLs: {successful_urls}")

        # Flatten the list of successful URLs
        successful_urls = [url for sublist in all_successful_urls for url in sublist]

        # Log all successful URLs
        logging.info("All successful URLs %s:",successful_urls)

        # Prepare the message for azure_stream
        prompt_messages = message_prompt(
            original_query, combined_context, format_option, messages
        )

        # Return both the streaming response and the weaviate_context data
        response_data = {
            "internet_links": successful_urls,
        }

        response_headers = {
            "response_data": json.dumps(response_data)
        }

        end_time = time.time()
        total_time = end_time - start_time
        logging.info("Total processing time for internet search: %.2f seconds", total_time)

        return StreamingResponse(azure_stream(prompt_messages), media_type="text/event-stream",headers=response_headers)
    except Exception as error:
        end_time = time.time()
        total_time = end_time - start_time
        logging.error("Error in internet_search (after %.2f seconds): %s", total_time, str(error))
        return JSONResponse(content={"error": str(error)}, status_code=500)

@app.post("/gpt-search")
async def gpt_search(request: Request, api_key: APIKey = Depends(get_api_key)):
    """
    Handles the POST request for chat with GPT.

    Args:
        request (Request): The FastAPI request object.

    Returns:
        StreamingResponse: The streaming response containing the generated content.

    Raises:
        JSONResponse: If there is an error during the request.

    """
    try:
        data = await request.json()
        query = data.get("query", "")
        messages = data.get("messages", [])
        if not query:
            raise HTTPException(status_code=400, detail="query is required.")
        message = {"role": "system", "content": query}
        messages.append(message)
        return StreamingResponse(azure_stream(messages), media_type="text/event-stream")
    except Exception as error:
        return JSONResponse(content={"error": str(error)}, status_code=500)

async def file_processing_stream(file_content: bytes, file_name: str, session_id: str, target_count: int = 7):
    """
    Process the file content by extracting pages and tables, creating chunks, and storing them in a vector database.

    Args:
        file_content (bytes): The content of the file to be processed.
        file_name (str): The name of the file.
        session_id (str): The session ID associated with the file.

    Yields:
        str: Status updates and error messages in the form of JSON strings.

    Returns:
        None
    """
    start_time = time.time()  # Start timing

    try:
        yield f'data: {{"status": "Extracting pages and tables for {file_name}"}}\n\n'
        content = await extract_pages_tables(file_content)

        all_chunks = []
        yield f'data: {{"status": "Creating chunks for {file_name}"}}\n\n'
        if isinstance(content, list):
            if not content:
                yield f'error: {{"status": "file is not processed {file_name}"}}\n\n'
                return
            for page in content:
                page_chunks = recursive_chunks(page)
                all_chunks.extend(page_chunks)
            
        else:
            yield f'data: {{"status": "Creating chunks for {file_name}"}}\n\n'
            print(f"Creating chunks for {file_name}")
            all_chunks = recursive_chunks(content)

        yield f'data: {{"status": "getting NER tags {file_name}"}}\n\n'

        if EXTRACT_NERTAGS != None:
            if EXTRACT_NERTAGS.lower() == 'true':
                if isinstance(content, str):
                    content = recursive_chunks(content, nertags=True)
                ner_tags = await get_and_trim_entities(content, target_count=target_count)
                if 'error' in  ner_tags:
                    yield f'error: {{"Error in extracting NER tags: {ner_tags["error"]}"}}\n\n'
                    return
                if 'no_data' in ner_tags:
                    yield f'status: {ner_tags["no_data"]}"\n\n'
                else:
                    yield f'data: {{"status": "found NER tags {len(json.loads(ner_tags))}"}}\n\n'

        yield f'data: {{"status": "Storing chunks to vector database for {file_name}"}}\n\n'
        weaviate_client = await get_weaviate_client()
        result = await add_pages_to_document(
            weaviate_client, all_chunks, session_id, file_name, ner_tags
        )

        if "success" in result:
            yield f'data: {{"status": "File {file_name} processed and saved successfully"}}\n\n'
        else:
            yield f'data: {{"status": "Error processing file {file_name}"}}\n\n'
    except Exception as e:
        logging.error("Error handling the file upload: %s", str(e))
        yield f'data: {{"error": "{str(e)}"}}\n\n'
    
    end_time = time.time()  # End timing
    total_time = end_time - start_time
    logging.info(f"File processing stream completed in {total_time} seconds")

@app.post("/process-file")
async def process_file_endpoint(
    session_id: str = Form(..., description="A unique session identifier"),
    file: UploadFile = File(..., description="file to be processed"),
    api_key: APIKey = Depends(get_api_key),
):
    """
    Process a file asynchronously and return a streaming response.

    Args:
        session_id (str): A unique session identifier.
        file (UploadFile): The file to be processed.
        api_key (APIKey): The API key for authentication.

    Returns:
        StreamingResponse: A streaming response with the processed file content.

    Raises:
        HTTPException: If the file or session_id is missing.
        JSONResponse: If there is an error handling the file upload.
    """
    if not file or not session_id:
        raise HTTPException(status_code=400, detail="file and session_id are required.")
    try:
        start_time = time.time()  # Start timing

        file_content = await file.read()  # Read file content once at the beginning

        async def event_generator():
            async for message in file_processing_stream(
                file_content, file.filename, session_id
            ):
                yield message
                await asyncio.sleep(0.05)  # Reduce sleep time to ensure quicker processing

        end_time = time.time()  # End timing
        total_time = end_time - start_time
        logging.info(f"File processing completed in {total_time} seconds")

        return StreamingResponse(event_generator(), media_type="text/event-stream")
    except Exception as e:
        logging.error("Error handling the file upload: %s", str(e))
        return JSONResponse(content={"error": str(e)}, status_code=500)
    
@app.post("/upload-search")
async def search_document(request: Request, api_key: APIKey = Depends(get_api_key)):
    """
    Handle a POST request to search documents using Weaviate and optionally the internet.

    Args:
        request (Request): The incoming HTTP request containing the search query and options.
        api_key (APIKey, optional): API key for authentication.

    Returns:
        StreamingResponse: A streaming response containing the search results.

    Raises:
        HTTPException: If the 'query' or 'session_id' fields are missing in the request body.
        JSONResponse: If an error occurs during processing.
    """
    try:
        data = await request.json()
        query = data.get("query", "")
        session_id = data.get("session_id", "")
        format_option = data.get("format_option", "")
        messages = data.get("messages", [])
        internet_rag = data.get("internet_rag", False)
        num_alternative = data.get("num_alternatives", num_alternatives)

        if not query or not session_id:
            raise HTTPException(
                status_code=400, detail="query and session_id are required."
            )

        alternative_questions = await generate_alternative_questions(
            query, num_alternative, format_option
        )
        
        if not alternative_questions:
            logging.warning("No alternative questions were generated. Using original query.")
            alternative_questions = [query]
        
        logging.info("Questions to be processed: %s", alternative_questions)

        weaviate_client = await get_weaviate_client()
        all_chunks = []
        all_successful_urls = []

        weaviate_client = await get_weaviate_client()
        all_successful_urls =[]
        async def process_query(alt_query):

            weaviate_chunks = await retrieve_session_data(
                weaviate_client, session_id, alt_query
            )
            internet_context = None

            if weaviate_chunks:
                logging.info(
                    f"Retrieved {len(weaviate_chunks)} chunks for query: '{alt_query}'"
                )
                all_chunks.extend(weaviate_chunks)

            if internet_rag:
                connector = TCPConnector(ssl=False)
                async with ClientSession(connector=connector) as session:

                    internet_context, successful_urls = await internet_search_async(
                        alt_query, session_id, session
                    )
                    all_successful_urls.extend(successful_urls)
                logging.info(
                    f"Internet context for '{alt_query}': {internet_context[:100]}..."
                )

            return internet_context

        # Process all queries concurrently
        internet_contexts = await asyncio.gather(
            *[process_query(q) for q in alternative_questions]
        )

        # Deduplicate chunks
        unique_chunks = {chunk['id']: chunk['content'] for chunk in all_chunks}
        logging.info(f"Total unique chunks after deduplication: {len(unique_chunks)}")

        combined_weaviate_context = "\n\n".join(unique_chunks.values())
        combined_internet_context = "\n\n".join(filter(None, internet_contexts)) if internet_rag else ""

        if internet_rag:
            context = with_internet_context(
                combined_weaviate_context, combined_internet_context
            )
        else:
            context = combined_weaviate_context

        prompt_messages = message_prompt(query, context, format_option, messages)
        response_data = {
            "internet_links": all_successful_urls,
        }

        response_headers = {
            "response_data": json.dumps(response_data)
        }

        return StreamingResponse(azure_stream(prompt_messages), media_type="text/event-stream",headers=response_headers)
    except Exception as error:
        logging.error(f"Error in /upload-search: {str(error)}")
        return JSONResponse(content={"error": str(error)}, status_code=500)


@app.post("/deletefiles")
async def delete_files(request: Request, api_key: APIKey = Depends(get_api_key)):
    """
    Deletes a file from the vector database and streams the status.

    Returns:
        StreamingResponse: A streaming response containing the status of the file deletion process.
    """
    try:
        data = await request.json()
        session_id = data.get("session_id", "")
        file_name = data.get("file_name", "")
        if not file_name or not session_id:
            raise HTTPException(
                status_code=400, detail="file_name and session_id are required."
            )
        
        async def deletion_stream():
            weaviate_client = await get_weaviate_client()
            yield f'data: {{"status": "Removing {file_name}"}}\n\n'
            result = await delete_weaviate_object(weaviate_client, session_id, file_name)
            yield f'data: {{"status": "{result}"}}\n\n'

        return StreamingResponse(deletion_stream(), media_type="text/event-stream")

    except Exception as error:
        return JSONResponse(content={"error": str(error)}, status_code=500)


@app.post("/connect-user-documents")
async def connect_user_documents(request: Request, api_key: APIKey = Depends(get_api_key)):
    """
    Connects user documents to their profile.

    Args:
        request (Request): The incoming HTTP request.

    Returns:
        dict: A dictionary containing the success message.

    Raises:
        HTTPException: If the request is invalid or fails to create/retrieve the user.
    """
    data = await request.json()
    logger.info(f"Received request data: {data}")

    user_id = data.get("user_id", "")
    file_ids = data.get("file_ids", [])  

    if not user_id or not file_ids:
        logger.error("Invalid request: 'user_id' or 'file_ids' missing")
        raise HTTPException(status_code=400, detail="Invalid request: 'user_id' or 'file_ids' missing")

    logger.info(f"Processing request for user_id: {user_id} with file_ids: {file_ids}")

    user_uuid = await create_user_details(weaviate_client, user_id)
    if not user_uuid:
        logger.error(f"Failed to create or retrieve user for user_id {user_id}")

    semaphore = asyncio.Semaphore(5)  # Limit to 5 concurrent tasks
    tasks = [process_file(weaviate_client, file_id, user_uuid, semaphore) for file_id in file_ids]

    await asyncio.gather(*tasks)

    logger.info(f"Successfully processed request for user_id: {user_id}")
    return {"message": "Connections added successfully"}

@app.post("/delete-userinfo")
async def delete_userinfo(request: Request, api_key: APIKey = Depends(get_api_key)):
    """
    Delete user information by user ID.

    Args:
        request (Request): The incoming request object.

    Returns:
        dict: A dictionary containing a success message if the user information is deleted successfully.

    Raises:
        HTTPException: If the user information with the specified user ID is not found or if there is an internal server error.
    """
    data = await request.json()
    user_id = data.get("user_id", "")
    try:
        success = await delete_userinfo_by_user_id(weaviate_client, user_id)
        if not success:
            raise HTTPException(status_code=404, detail=f"UserInfo with user_ID {user_id} not found.")
        return {"message": f"UserInfo with user_ID {user_id} deleted successfully."}
    except Exception as ex:
        logger.error(f"Error deleting UserInfo for user_id {user_id}: {ex}")
        raise HTTPException(status_code=500, detail="Internal server error.")

def message_prompt(prompt, context, format_option, messages):
    """
    Appends a system message to the given list of messages.

    Args:
        prompt (str): The question prompt.
        context (str): The context for the question.
        format_option (str): The desired format for the response.
        messages (list): The list of messages to append to.

    Returns:
        list: The updated list of messages.
    """
    messages.append(
            {
                "role": "system",
                "content": f"You will be given a context document, Format, chat history (if possible) and a question."
                f" Chat history will contain the previous questions and answers."
                f" Your task is to answer the question based solely on the information "
                f"provided in the context document. Do not use any external knowledge."
                f" Format: {format_options(format_option)}. Analyse the given context "
                f"properly to answer the question accurately: "
                f"<context><document>{context}</document></context>"
                f"question: {prompt}",
            }
        )
    return messages


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
