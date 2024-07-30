''' This file contains the common code used in the application. '''
import os
import json
import logging
import dotenv
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential


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


# NER Code
# Azure TextAnalytics auth credentials
async def authenticate_client(key, endpoint):
    """
    Authenticates the client using the provided key and endpoint.

    Parameters:
    - key (str): The Azure Text Analytics API key.
    - endpoint (str): The endpoint URL for the Text Analytics service.

    Returns:
    - text_analytics_client (TextAnalyticsClient): The authenticated TextAnalyticsClient object.

    """
    ta_credential = AzureKeyCredential(key)
    text_analytics_client = TextAnalyticsClient(
        endpoint=endpoint,
        credential=ta_credential)
    return text_analytics_client

# Filter categories variables
pii_exclude_categories = ["Person", "PersonType", "PhoneNumber", "Address",
                        "Email", "URL", "IPAddress", "DateTime", "Quantity"]
normal_include_categories = ["Organization", "Product", "Location"]

# Extract NER tags
async def get_ner_tags(splitted_contents):
    """
    Extracts Named Entity Recognition (NER) tags from the given list of contents.

    Args:
        splitted_contents (list): A list of strings representing the contents 
        to extract NER tags from.

    Returns:
        dict: A dictionary containing the extracted NER tags as key-value pairs, 
        where the key is the entity text
              and the value is the entity category.

    Raises:
        Exception: If there is an error while extracting NER tags.

    Example:
        splitted_contents = ["I live in New York.", "My email is john@example.com"]
        ner_tags = await get_ner_tags(splitted_contents)
        print(ner_tags)
        # Output: {'New York': 'Location', 'john@example.com': 'EmailAddress'}
    """
    if not splitted_contents:
        return {}
    # Authenticate client
    client = await authenticate_client(TEXT_ANALYTICS_AZURE_KEY, TEXT_ANALYTICS_AZURE_ENDPOPINT)
    ner_extracted_entities = {}
    try:
        # Extract NER entities from the splitted contents by Filter categories variables
        def extract_ner_entities(content):
            ner_entities = {}
            pii_tags = client.recognize_pii_entities(documents=[content])[0]
            normal_tags = client.recognize_entities(documents=[content])[0]
            for pii_entity in pii_tags.entities: # Iterate over PII entities
                if pii_entity.category not in pii_exclude_categories: # Exclude PII categories
                    if pii_entity.text not in ner_entities: # Avoid duplicates
                        ner_entities[pii_entity.text] = pii_entity.category
            for normal_entity in normal_tags.entities: # Iterate over PII entities
                if normal_entity.category in normal_include_categories: # Include normal categories
                    if normal_entity.text not in ner_entities: # Avoid duplicates
                        ner_entities[normal_entity.text] = normal_entity.category
            return ner_entities
        # Handle chunks length more than 5120 characters by splitting them into smaller chunks
        for content in splitted_contents:
            if len(content) > 5120:
                content_len_div = len(content)//5120
                if len(content)/5120 > content_len_div:
                    content_len_div += 1
                split_list = content.split(" ")
                start_index = 0
                step_size = len(split_list)//content_len_div
                for num in range(0, len(split_list), step_size):
                    split_list_str = " ".join(split_list[start_index:start_index+step_size])
                    entities = extract_ner_entities(split_list_str)
                    ner_extracted_entities.update(entities)
                    start_index += step_size
                    num += step_size
            else:
                ner_extracted_entities.update(extract_ner_entities(content))
    except Exception as e:
        logging.error('Error while extracting Nertags: %s', e)
        return {'error': str(e)}
    return ner_extracted_entities

async def get_and_trim_entities(contents, target_count=7):
    """
    Retrieves Named Entity Recognition (NER) tags from the given list of contents and 
    trims the entities based on the target count.

    Args:
        contents (list): A list of strings representing the contents to extract NER tags from.
        target_count (int, optional): The maximum number of entities to include for each category. 
        Defaults to 7.

    Returns:
        str: A JSON string containing the trimmed entities and their categories.

    Raises:
        Exception: If there is an error while retrieving or trimming the entities.

    Example:
        contents = ["I live in New York.", "My email is john@example.com"]
        trimmed_entities = await get_and_trim_entities(contents)
        print(trimmed_entities)
        # Output: {"New York": "Location", "john@example.com": "EmailAddress"}
    """
    new_data = {}
    data = await get_ner_tags(contents)
    if 'error' in data:
        return data
    if not data:
        return {'no_data': 'No important entities found in the given data.'}
    try:
        category_values = set(data.values())
        category_buckets = {
            category: [] for category in category_values
            if category in normal_include_categories
        }
        for item in data:
            category = data[item]
            if category in normal_include_categories:
                if len(category_buckets[category]) < target_count:
                    category_buckets[category].append(item)
                    new_data[item] = category
            if category not in category_buckets:
                new_data[item] = category
        return json.dumps(new_data, indent=2)
    except Exception as e:
        logging.error('Error while trimming entities: %s', e)
        return json.dumps({'error': str(e)}, indent=2)

# Prompt format Code
prompt_format = {}
prompt_format["EXECUTIVE_SUMMARY_PROMPT"] = """
You are tasked with creating a comprehensive and professional executive summary based on a given document. Your goal is to distill the key information into a well-structured, informative summary that covers essential aspects of the business deal, company overview, financials, and investment details.

You will be provide document in this format:

<document>
Here will be the content of the document.
</document>

Carefully read and analyze the provided document. Pay close attention to key details, financial figures, and strategic information. As you review the content, mentally organize the information into relevant categories that align with the sections of a typical executive summary.

Generate an executive summary that covers the following sections, to the extent that information is available in the document:

1. Executive Summary
2. Deal Overview
3. Business Overview
4. Financial History and Forecast
5. Pricing
6. Deal Structure
7. Base Case and Returns
8. Investment Thesis
9. Key Risks

For each section:
- Include all relevant information found in the document
- If specific information for a section is not available, provide a brief statement indicating the lack of information or skip the section entirely if appropriate
- Use clear, concise language suitable for a professional business document
- Present numerical data in a clear, easy-to-understand format
- Generate tables where appropriate to summarize financial or structural information

Be flexible in your approach. Not all documents will contain information for every section. Focus on presenting the available information in the most effective and professional manner possible.

Your executive summary should be comprehensive and detailed, reflecting the complexity of the business deal or investment opportunity. Aim for a length that adequately covers all available information without unnecessary repetition.

Present your executive summary in human markdown format. This means:
- Use appropriate markdown syntax for headers, lists, tables, and emphasis
- Ensure proper nesting of headers (e.g., ## for main sections, ### for subsections)
- Use bullet points or numbered lists where appropriate
- Create tables using markdown syntax for financial or structural information
- Apply bold or italic formatting to highlight key points or figures
- Strictly follow these and present in human markdown format

Remember to maintain a professional tone throughout the summary and ensure that the information is presented in a logical, easy-to-follow structure.
"""

prompt_format["KEY_RISKS_PROMPT"] = """
You are tasked with creating a comprehensive key risks section based on a given document. Your goal is to identify and analyze potential risks associated with the business or investment opportunity described in the document.

You will be provided with a document in this format:

<document>
Here will be the content of the document.
</document>

Carefully read and analyze the provided document. Pay close attention to any information related to risks, challenges, or potential obstacles faced by the business or investment opportunity.

Generate a key risks section that covers the following categories and subcategories, to the extent that information is available in the document:

1. Market Conditions
   - Economic Environment
   - Consumer Demand
   - Market Positioning

2. Customer / Contract Concentration
   - Concentration Risks
     - Key Contracts
     - Impact of Losing Key Contracts
   - Long-Term Partnerships and Quality Track Record
     - Partnership Stability
     - Quality Performance

3. Industry Dynamics
   - Pace of Change
     - Market Disruptions
   - Opportunities Amidst Change
     - Growth Trends
     - Management Adaptability

4. Regulatory and Policy Risks
   - Regulatory Changes
     - Impact of Policy Changes
   - Compliance

5. Operational Risks
   - Leadership Changes
     - Key Personnel
   - Operational Efficiency
     - Efficiency Challenges
   - Supply Chain Management
     - Supply Chain Risks

6. Competitive Risks
   - Market Competition
     - Competitive Landscape
   - Quality vs. Cost

For each category and subcategory:
- Include all relevant information found in the document
- If specific information for a category or subcategory is not available, provide a brief statement indicating the lack of information or skip it entirely if appropriate
- Use clear, concise language suitable for a professional business document
- Present any numerical data in a clear, easy-to-understand format

Be flexible in your approach. Not all documents will contain information for every category or subcategory. Focus on presenting the available risk information in the most effective and professional manner possible.

Your key risks section should be comprehensive and detailed, reflecting the complexity of the business or investment opportunity. Aim for a length that adequately covers all available risk information without unnecessary repetition.

Present your key risks section in human markdown format. This means:
- Use appropriate markdown syntax for headers, lists, and emphasis
- Ensure proper nesting of headers (e.g., ## for main categories, ### for subcategories)
- Use bullet points where appropriate
- Apply bold or italic formatting to highlight key points or figures
- Make sure that raw markdown format is not generated

If there is insufficient information to populate a particular risk category or subcategory, you may omit it or provide a brief statement indicating the lack of information.

Remember to maintain a professional tone throughout the key risks section and ensure that the information is presented in a logical, easy-to-follow structure.
"""

prompt_format["MARKET_OVERVIEW_PROMPT"] = """
You are tasked with creating a comprehensive and professional market overview based on a given document. Your goal is to distill the key information into a well-structured, informative overview that covers essential aspects of the global market environment, industry landscape, competitive analysis, consumer trends, technological advancements, regulatory environment, regional market analysis, and market opportunities and challenges.

You will be provided with a document in this format:

<document>
Here will be the content of the document.
</document>

Carefully read and analyze the provided document. Pay close attention to key details, market figures, and strategic information. As you review the content, mentally organize the information into relevant categories that align with the sections of a typical market overview.

Generate a market overview that covers the following sections, to the extent that information is available in the document:

1. Global Market Environment
   - Economic Indicators
   - Geopolitical Factors
2. Industry Landscape
   - Market Size and Growth
   - Key Market Segments
3. Competitive Analysis
   - Major Players
   - Competitive Dynamics
4. Consumer Trends
   - Consumer Preferences
   - Demand Drivers
5. Technological Advancements
   - Emerging Technologies
   - Innovation Trends
6. Regulatory Environment
   - Current Regulations
   - Future Regulatory Changes
7. Regional Market Analysis
   - North America
   - Europe
   - Asia-Pacific
8. Market Opportunities and Challenges
   - Growth Opportunities
   - Market Challenges

For each section:
- Include all relevant information found in the document
- If specific information for a section is not available, provide a brief statement indicating the lack of information or skip the section entirely if appropriate
- Use clear, concise language suitable for a professional business document
- Present numerical data in a clear, easy-to-understand format
- Generate tables where appropriate to summarize market or structural information

Be flexible in your approach. Not all documents will contain information for every section. Focus on presenting the available information in the most effective and professional manner possible.

Your market overview should be comprehensive and detailed, reflecting the complexity of the market landscape. Aim for a length that adequately covers all available information without unnecessary repetition.

Present your market overview in human markdown format. This means:
- Use appropriate markdown syntax for headers, lists, tables, and emphasis
- Ensure proper nesting of headers (e.g., ## for main sections, ### for subsections)
- Use bullet points or numbered lists where appropriate
- Create tables using markdown syntax for market or structural information
- Apply bold or italic formatting to highlight key points or figures

Remember to maintain a professional tone throughout the overview and ensure that the information is presented in a logical, easy-to-follow structure.
"""
prompt_format["DEFAULT_PROMT"] ="""
Analyze the provided document context and determine its relevance to the question. Include all relevant information found in the document, summarizing key points in clear, concise markdown format. For any requested information not present in the document, clearly state its absence. If no relevant information is found for any part of the question, indicate that the document lacks information on all requested topics."""

internet_prompt = """
# IDENTITY and PURPOSE

Please be brief. Combine both Internet Context and Knowledgebase Context to provide high quality answer.

# STEPS

Please be brief. Combine both Internet Context and Knowledgebase Context to provide high quality answer.

In some cases, Other context will also be provided which is basically the chat history. This is also a part of context, so make sure you don't skip this fact.

You goal is to understand the context carefully and provide the best possible answer.

# OUTPUT INSTRUCTIONS
Understand the question and then answer.

INPUT: The question asked by the user.
"""

def format_options(key: str = "DEFAULT_PROMT"):
    """
    Returns the prompt format based on the given key.

    Parameters:
    key (str): The key to retrieve the prompt format. Defaults to "DEFAULT_PROMT".

    Returns:
    str: The prompt format corresponding to the given key.
    """
    return prompt_format[key] if key in prompt_format else prompt_format["DEFAULT_PROMT"]

def with_internet_context(weaviate_context: str, internet_context: str):
    """
    Generates a prompt with the user query, internet context, and Knowledgebase context.

    Parameters:
    user_query (str): The user's query.
    Knowledgebase_context (str): The context retrieved from Knowledgebase.
    internet_context (str): The context retrieved from the internet.

    Returns:
    str: The generated prompt.
   """
    retrieved_and_internet_prompt = f"""
    Internet Context:\n
    {internet_context}\n

    Knowledgebase Context:\n
    {weaviate_context}\n

    {internet_prompt}
    """
    return retrieved_and_internet_prompt
