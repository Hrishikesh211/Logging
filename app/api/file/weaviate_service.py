'''Weaviate Storage Module'''

import asyncio
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_community.document_loaders import PyPDFLoader
from openai import AzureOpenAI
from weaviate import Client as weaviateClient
from weaviate.auth import AuthApiKey
from app.common.common_code import (
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_ENDPOINT,
    OPENAI_API_VERSION,
    DOCINTELLIGENCE_ENDPOINT,
    EMBEDDING_MODEL,
    DOCINTELLIGENCE_KEY,
    WEAVIATE_API_KEY,
    WEAVIATE_URL,
)
from fastapi import HTTPException
import io
import logging
from pypdf import PdfReader,PdfWriter


_weaviate_client = None
_weaviate_client_lock = asyncio.Lock()
# Configure logging
logging.basicConfig(level=logging.DEBUG)
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def get_weaviate_client():
    """
    Retrieves or creates a Weaviate client instance and returns it.
    If the Weaviate client instance doesn't exist, it creates a new instance
    using the specified Weaviate URL and API key. The created instance is then
    stored in the global variable _weaviate_client for future use.

    Returns:
        The Weaviate client instance.
    """
    global _weaviate_client
    if _weaviate_client is None:
        async with _weaviate_client_lock:
            if _weaviate_client is None:
                try:
                    _weaviate_client = weaviateClient(
                        url=WEAVIATE_URL,
                        auth_client_secret=AuthApiKey(api_key=WEAVIATE_API_KEY),
                        timeout_config=(5, 1000)
                    )
                    if not _weaviate_client.is_live():
                        raise Exception("Weaviate client is not live")
                except Exception as e:
                    logging.error(f"Failed to create Weaviate client: {e}")
                    raise HTTPException(status_code=500, detail="Weaviate client initialization failed")
    return _weaviate_client

embedding_client = AzureOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    api_version=OPENAI_API_VERSION,
)

def split_uploaded_file_into_pages(uploaded_file_content):
    """
    Splits the uploaded file content into individual pages and returns a list of page contents.

    Args:
        uploaded_file_content (bytes): The content of the uploaded file.

    Returns:
        list: A list of page contents, where each element represents the content of a single page.
    """
    pdf_reader = PdfReader(io.BytesIO(uploaded_file_content))
    pages = []

    for page_num in range(len(pdf_reader.pages)):
        pdf_writer = PdfWriter()
        pdf_writer.add_page(pdf_reader.pages[page_num])
        
        page_bytes = io.BytesIO()
        pdf_writer.write(page_bytes)
        pages.append(page_bytes.getvalue())

    return pages

def extract_text_pypdf(pdf_content):
    """
    Extracts text from a PDF using PyPDF2 library.
    Args:
        pdf_content (bytes): The content of the PDF file.

    Returns:
        str: The extracted text from the PDF.

    Raises:
        Exception: If there is an error while extracting text from the PDF.

    """
    try:
        pdf_reader = PdfReader(io.BytesIO(pdf_content))
        text = ""
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
        return text
    except Exception as e:
        print(f"Error extracting text from PDF using PyPDF2: {e}")
        return ""

async def extract_pages_tables(uploaded_file_content):
    """
    Extracts tables from each page of an uploaded file using document intelligence.

    Args:
        uploaded_file_content (bytes): The content of the uploaded file.

    Returns:
        List[str]: A list of extracted tables from each page of the file.

    Raises:
        Exception: If there is an error in the document intelligence process.

    """
    semaphore = asyncio.Semaphore(30)  # Increased concurrency

    async def analyze_page(page_content):
        async with semaphore:
            try:
                document_intelligence_client = DocumentIntelligenceClient(
                    endpoint=DOCINTELLIGENCE_ENDPOINT,
                    credential=AzureKeyCredential(DOCINTELLIGENCE_KEY),
                )

                poller = await asyncio.to_thread(
                    document_intelligence_client.begin_analyze_document,
                    model_id="prebuilt-layout",
                    analyze_request=page_content,
                    content_type="application/octet-stream",
                )
                result = await asyncio.to_thread(poller.result)
                return result.content
            except Exception as e:
                print(f"Error in document intelligence: {e}")
                return extract_text_pypdf(page_content)

    try:
        pages = split_uploaded_file_into_pages(uploaded_file_content)
        tasks = [analyze_page(page) for page in pages]
        results = await asyncio.gather(*tasks)
        return results
    except Exception as e:
        print(f"Error in document intelligence: {e}")
        return extract_text_pypdf(uploaded_file_content)

# Create recursive chunks
def recursive_chunks(content, nertags=False):
    """
    Splits the given content into smaller chunks using a recursive character text splitter.

    Args:
        content (str): The content to be split into chunks.

    Returns:
        list: A list of smaller chunks created from the content.
    """
    md_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.MARKDOWN, chunk_size=2000, chunk_overlap=300
    )
    if nertags:
        return md_splitter.split_text(content)
    return md_splitter.create_documents([content])

# Add pages to document in Weaviate
async def add_pages_to_document(weaviate_client, pages, session_id, file_name, ner_tags):
    """
    Add pages to a document in Weaviate.

    Args:
        weaviate_client (WeaviateClient): The Weaviate client.
        pages (List[str]): The list of pages to be added.
        session_id (str): The session ID.
        file_name (str): The name of the file.

    Returns:
        dict: A dictionary indicating the success or failure of the operation.

    Raises:
        Exception: If there is an error adding page data.

    """
    semaphore = asyncio.Semaphore(20)  # Increased concurrency

    async def process_page(page_content):
        async with semaphore:
            try:
                if ner_tags:
                  nertags = ner_tags.replace('\n ','').replace('\n','').replace(',','')\
                      .replace(': ','=').replace(':','=')
                  page_content = nertags + str(page_content)
                vector = await asyncio.to_thread(
                    embedding_client.embeddings.create,
                    model=EMBEDDING_MODEL,
                    input=[str(page_content)],
                )
                embedding_object = vector.data[0]
                vector = embedding_object.embedding
                if vector:
                    client_data_object = {
                        "session_id": session_id,
                        "data": str(page_content),
                        "file_name": file_name,
                    }
                    await asyncio.to_thread(
                        weaviate_client.data_object.create,
                        data_object=client_data_object,
                        class_name="SessionData",
                        vector=vector,
                    )
            except Exception as e:
                print(f"Error adding page data: {e}")
                return {"error": "Error adding page data"}

    tasks = [process_page(page) for page in pages]
    await asyncio.gather(*tasks)
    return {"success": "Pages added successfully"}

def combine_documents_to_string(documents):
    """
    Combines the content of multiple documents into a single string.

    Args:
        documents (list): A list of documents.

    Returns:
        str: The combined content of the documents.
    """
    return " ".join(doc.page_content for doc in documents)


async def retrieve_session_data(weaviate_client, session_id, query_text):
    """
    Retrieve session data from Weaviate based on the session ID and query text.

    Args:
        weaviate_client (WeaviateClient): The Weaviate client object.
        session_id (str): The session ID to filter the data.
        query_text (str): The query text to search for.

    Returns:
        list: A list of dictionaries, each containing 'id' and 'content' of the retrieved chunks.

    Raises:
        None
    """
    try:
        # Generate the vector for the query text
        vector = await asyncio.to_thread(
            embedding_client.embeddings.create, model=EMBEDDING_MODEL, input=query_text
        )

        embedding_object = vector.data[0]
        vector = embedding_object.embedding

        # Perform a hybrid search using both the query text and its vector representation
        result = await asyncio.to_thread(
            weaviate_client.query.get("SessionData", ["data", "session_id"])
            .with_additional(["id"])  # Request the Weaviate object ID
            .with_hybrid(query=query_text, vector=vector)
            .with_where(
                {"path": ["session_id"], "operator": "Equal", "valueString": session_id}
            )
            .with_limit(3)
            .do
        )

        # Extract and return the 'id' and 'data' fields from each entry in the results
        if (
            "data" in result
            and "Get" in result["data"]
            and "SessionData" in result["data"]["Get"]
        ):
            return [
                {"id": entry["_additional"]["id"], "content": entry["data"]}
                for entry in result["data"]["Get"]["SessionData"]
            ]
        print("No data returned from the query.")
        return []
    except Exception as e:
        print(f"Error retrieving session data: {e}")
        return []


async def delete_weaviate_object(weaviate_client, session_id, file_name):
    """
    Deletes a Weaviate object based on the session ID and file name.

    Args:
        weaviate_client (WeaviateClient): The Weaviate client object.
        session_id (str): The session ID of the object to be deleted.
        file_name (str): The file name of the object to be deleted.

    Returns:
        dict: A dictionary containing the result of the deletion operation.
    """
    delete_result = {}
    try:
        result = await asyncio.to_thread(
            weaviate_client.batch.delete_objects,
            class_name="SessionData",
            where={
                "operator": "And",
                "operands": [
                    {
                        "path": ["file_name"],
                        "operator": "Equal",
                        "valueString": file_name,
                    },
                    {
                        "path": ["session_id"],
                        "operator": "Equal",
                        "valueString": session_id,
                    },
                ],
            },
        )
        if result["results"]["successful"] > 0 and result["results"]["failed"] == 0:
            delete_result.update({file_name: "deleted"})
        else:
            delete_result.update({file_name: "file not found"})
    except Exception as e:
        print(f"Error deleting object: {e}")
        delete_result.update({file_name: "file not found"})
    return delete_result

async def get_user_uuid(weaviate_client, user_id: str):
    """
    Retrieve the UUID of the UserInfo object for the given user ID.

    Args:
        weaviate_client (WeaviateClient): The Weaviate client instance.
        user_id (str): The user ID for which to retrieve the UUID.

    Returns:
        str: The UUID of the UserInfo object for the given user ID, or None if not found.

    """
    try:
        print(f"Attempting to retrieve UUID for user_id: {user_id}")
        result = weaviate_client.query.get(
            "UserInfo",
            ["_additional { id }", "user_ID"]
        ).with_where({
            "operator": "Equal",
            "path": ["user_ID"],
            "valueString": user_id
        }).with_limit(1).do()
        
        user_info = result.get('data', {}).get('Get', {}).get('UserInfo', [])
        if not user_info:
            print(f"No UserInfo found for user_id: {user_id}")
            return None

        user_uuid = user_info[0].get('_additional', {}).get('id')
        print(f"Retrieved user_uuid: {user_uuid} for user_id: {user_id}")
        return user_uuid
    except Exception as ex:
        print(f"Error retrieving user UUID for user_id {user_id}: {ex}")
        return None

async def create_user_details(weaviate_client, user_ID: str):
    """
    Creates user documents with details in Weaviate for each file UUID.

    Args:
        weaviate_client: The Weaviate client object.
        user_ID (str): The ID of the user.

    Returns:
        str: The UUID of the created user.

    Raises:
        Exception: If there is an error creating user details.

    """
    user_uuid = await get_user_uuid(weaviate_client, user_ID)
    if user_uuid:
        print(f"User UUID already exists for user_ID {user_ID}: {user_uuid}")
        return user_uuid

    client_data_object = {
        "user_ID": user_ID,
    }

    try:
        print(f"Creating user details for user_ID: {user_ID}")
        create_response = weaviate_client.data_object.create(
            data_object=client_data_object, class_name="UserInfo"
        )
        user_uuid = (
            create_response.get("id")
            if isinstance(create_response, dict)
            else create_response
        )
        print(f"Created user UUID: {user_uuid} for user_ID: {user_ID}")
    except Exception as e:
        print(f"Error creating user details for user_ID {user_ID}: {e}")
        return None  # Ensure consistent return type

    return user_uuid

async def get_document_uuid(weaviate_client, file_id):
    """
    Retrieves the UUID of a document given its file ID.

    Args:
        weaviate_client (WeaviateClient): The Weaviate client object.
        file_id (str): The file ID of the document.

    Returns:
        str: The UUID of the document, or None if not found.
    """
    try:
        print(f"Attempting to retrieve UUID for file_id: {file_id}")
        result = weaviate_client.query.get(
            "Document",
            ["file_id", "_additional { id }"]
        ).with_where({
            "path": ["file_id"],
            "operator": "Equal",
            "valueString": file_id
        }).with_limit(1).do()

        print(f"Query result for file_id {file_id}: {result}")
        
        documents = result.get('data', {}).get('Get', {}).get('Document', [])
        if not documents:
            print(f"No Document found for file_id: {file_id}")
            return None
        
        document_uuid = documents[0].get('_additional', {}).get('id')
        print(f"Retrieved document UUID: {document_uuid} for file_id: {file_id}")
        return document_uuid
    except Exception as ex:
        print(f"Error retrieving document UUID for file_id {file_id}: {ex}")
        return None


async def add_connection_between_user_and_document(weaviate_client, document_uuid, user_uuid):
    """Adds a connection between the user and the document.

    Args:
        weaviate_client (WeaviateClient): The Weaviate client object.
        document_uuid (str): The UUID of the document.
        user_uuid (str): The UUID of the user.

    Raises:
        HTTPException: If there is an error adding the connection.

    """
    try:
        print(f"Adding connection between user_uuid: {user_uuid} and document_uuid: {document_uuid}")
        weaviate_client.data_object.reference.add(
            from_uuid=document_uuid,
            from_property_name="user_info",
            to_uuid=user_uuid,
            from_class_name="Document",
            to_class_name="UserInfo"
        )
        print(f"Successfully added connection for document_uuid: {document_uuid}")
    except Exception as ex:
        print(f"Error adding connection between user {user_uuid} and document {document_uuid}: {ex}")
  

async def process_file(weaviate_client, file_id, user_uuid, semaphore):
    """
    Process a file by creating or retrieving a document UUID and adding a connection between the user and the document.

    Args:
        weaviate_client (WeaviateClient): The Weaviate client object.
        file_id (str): The ID of the file to process.
        user_uuid (str): The UUID of the user.
        semaphore (asyncio.Semaphore): The semaphore to limit concurrent access.

    Raises:
        HTTPException: If failed to create or retrieve the document UUID.

    Returns:
        None
    """
    async with semaphore:
        document_uuid = await get_document_uuid(weaviate_client, file_id)
        if not document_uuid:
            print(f"Failed to create or retrieve Document UUID for {file_id}")
        await add_connection_between_user_and_document(weaviate_client, document_uuid, user_uuid)

async def delete_userinfo_by_user_id(weaviate_client, user_id):
    """Deletes a UserInfo object based on user_ID.

    Args:
        weaviate_client (WeaviateClient): The Weaviate client instance.
        user_id (str): The user ID of the UserInfo object to be deleted.

    Returns:
        bool: True if the UserInfo object was successfully deleted, False otherwise.
    """
    try:
        # Retrieve the UUID of the UserInfo object with the given user_ID
        user_uuid = await get_user_uuid(weaviate_client, user_id)
        if not user_uuid:
            print(f"UserInfo with user_ID {user_id} does not exist.")
            return False
        
        # Delete the UserInfo object using the UUID
        weaviate_client.data_object.delete(user_uuid, class_name="UserInfo")
        print(f"Deleted UserInfo with user_ID {user_id} and UUID {user_uuid}.")
        return True
    except Exception as ex:
        print(f"Error deleting UserInfo with user_ID {user_id}: {ex}")
        return False