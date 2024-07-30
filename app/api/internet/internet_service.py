"""Internet search and context retrieval functions for the RMB."""

import logging
import uuid
import io
import time
from urllib.parse import urlparse
import asyncio
from asyncio import TimeoutError
import tiktoken
import html2text
import chromadb
from bs4 import BeautifulSoup
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from app.common.common_code import BING_SEARCH_API_KEY, BING_SEARCH_ENDPOINT
from aiohttp import ClientError


def num_tokens_from_messages(message, model="gpt-4-32k-0613"):
    """
    Calculates the number of tokens in a given message using the specified model.

    Args:
        message (str): The input message.
        model (str, optional): The name of the model to use. Defaults to "gpt-4-32k-0613".

    Returns:
        int: The number of tokens in the message.

    Raises:
        KeyError: If the specified model is not found.
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    try:
        return len(encoding.encode(message))
    except Exception as e:
        print(f"Error encoding message: {e}")
        return 1000


async def get_top_5_urls_bing_search_async(query, session):
    """
    Retrieves the top 5 URLs from Bing search results for the given query.

    Args:
        query (str): The search query.
        session (aiohttp.ClientSession): The aiohttp client session.

    Returns:
        list: A list of the top 5 URLs.

    Raises:
        ClientError: If there is an error in the Bing search request.
    """
    mkt = "en-US"
    params = {"q": query, "mkt": mkt}
    headers = {"Ocp-Apim-Subscription-Key": BING_SEARCH_API_KEY}
    max_retries = 3
    retry_delay = 1

    for attempt in range(max_retries):
        try:
            async with session.get(
                BING_SEARCH_ENDPOINT, headers=headers, params=params, ssl=True
            ) as response:
                response.raise_for_status()
                json = await response.json()
                results = json["webPages"]["value"]
                if results:
                    return [result["url"] for result in results[:5]]
                logging.error("No results found for the given query.")
                return []
        except ClientError as e:
            if attempt < max_retries - 1:
                logging.warning(
                    f"Attempt {attempt + 1} failed. Retrying in {retry_delay} seconds..."
                )
                await asyncio.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                logging.error(f"Error in Bing search after {max_retries} attempts: {e}")
                return []


async def scrape_best_url_for_context_async(url, session_id, session):
    """
    Scrapes the best URL for context asynchronously.

    Args:
        url (str): The URL to scrape.
        session_id (int): The session ID.
        session: The session object.

    Returns:
        str: The scraped context.

    Raises:
        asyncio.TimeoutError: If a timeout occurs while scraping the URL.
        Exception: If an error occurs while scraping the URL.
    """
    context = ""

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
    }

    h = html2text.HTML2Text()
    h.ignore_links = False
    h.ignore_images = False
    h.ignore_tables = False

    try:
        async with session.get(
            url, headers=headers, timeout=10
        ) as response:  # 10-second timeout for initial request
            response.raise_for_status()
            content = await asyncio.wait_for(
                response.read(), timeout=10
            )  # 10-second timeout for reading content

            if urlparse(url).path.lower().endswith(
                ".pdf"
            ) or "application/pdf" in response.headers.get("Content-Type", ""):
                # Handle PDF
                pdf_reader = PdfReader(io.BytesIO(content))
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                    if num_tokens_from_messages(text) > 4000:
                        logging.info(
                            f"Token limit exceeded for PDF {url}, truncating content."
                        )
                        break
                context += f"# Content from PDF: {url}\n\n{text}\n\n"
            else:
                # Handle HTML
                soup = BeautifulSoup(content, "html.parser")
                for script in soup(["script", "style"]):
                    script.decompose()
                markdown_text = h.handle(str(soup))

                if num_tokens_from_messages(markdown_text) > 4000:
                    logging.info(f"Token limit exceeded for {url}, truncating content.")
                    markdown_text = markdown_text[:4000]  # Simple truncation
                context += markdown_text + "\n\n"

    except asyncio.TimeoutError:
        logging.warning(f"Timeout while scraping {url}")
    except Exception as e:
        logging.error(f"Error scraping {url}: {e}")

    return context


def recursive_chunks(content):
    """Splits the content into chunks of 2200 characters with 200 characters overlap"""
    md_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.MARKDOWN, chunk_size=2200, chunk_overlap=200
    )
    return md_splitter.split_text(content)


def context_chromadb(query, chunks, session_id):
    chroma_client = chromadb.Client()
    try:
        collection = chroma_client.get_or_create_collection(name=session_id)
        collection.add(
                documents = chunks,
                ids= [str(uuid.uuid4()) for _ in range(len(chunks))])
        collection = chroma_client.get_collection(name=session_id)
        results = collection.query(
            query_texts=[query],
            n_results=3 
        )
        doc_results = results['documents']
        final_context = " ".join([doc for doc in doc_results[0]])
    except Exception as e:
        logging.warning('Error processing chunks in chromadb', str(e))
        return ""
    # Deleting chroma db session
    try:
        chroma_client.delete_collection(name=session_id)
    except: #TODO Exception for client error not implemented
        logging.warning('Internet Rag chroma db session is not deleted')
    return final_context


async def internet_search_async(query, session_id, session):
    """
    Performs an asynchronous internet search using Bing and scrapes content from the top results.

    Args:
        query (str): The search query.
        session_id (str): The session ID for tracking purposes.
        session (ClientSession): The aiohttp ClientSession for making HTTP requests.

    Returns:
        tuple: A tuple containing the combined context from the scraped URLs and a list of successful URLs.

    Raises:
        TimeoutError: If a scraping operation times out.
        Exception: If an error occurs during scraping.
    """
    start_time = time.time()
    max_total_time = 20  # Maximum 20 seconds for the entire operation

    urls = await get_top_5_urls_bing_search_async(query, session)
    bing_search_time = time.time() - start_time
    logging.info(
        f"Bing search for '{query}' completed in {bing_search_time:.2f} seconds"
    )

    if not urls:
        logging.warning(f"No results found for query: {query}")
        return "No results found.", []

    combined_context = ""
    successful_urls = []
    for url in urls[:3]:  # Limit to first 3 URLs
        try:
            remaining_time = max_total_time - (time.time() - start_time)
            if remaining_time <= 0:
                logging.warning(f"Time limit exceeded for query: {query}")
                break

            context = await asyncio.wait_for(
                scrape_best_url_for_context_async(url, session_id, session),
                timeout=min(10, remaining_time),
            )
            if context.strip():  # Check if the context is not empty
                combined_context += context
                successful_urls.append(url)
                logging.info(f"Successfully scraped context from: {url}")

            if num_tokens_from_messages(combined_context) > 4000:
                logging.info(
                    f"Sufficient context gathered for '{query}', stopping further URL processing."
                )
                break
        except TimeoutError:
            logging.warning(f"Timeout while scraping {url}")
        except Exception as e:
            logging.error(f"Error scraping {url}: {e}")

    scraping_time = time.time() - start_time - bing_search_time
    logging.info(f"Web scraping for '{query}' completed in {scraping_time:.2f} seconds")

    if num_tokens_from_messages(combined_context) > 700:
        chunks = recursive_chunks(combined_context)
        combined_context = context_chromadb(query, chunks, session_id)

    total_time = time.time() - start_time
    logging.info(f"Total processing time for '{query}': {total_time:.2f} seconds")
    logging.info(f"Successful URLs: {successful_urls}") 
    return combined_context, successful_urls

