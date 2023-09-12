import os
import sys

from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pinecone
import tiktoken
from uuid import uuid4
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

from langchain.document_loaders import (
    CSVLoader,
    PDFMinerLoader,
    TextLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
)

from config import *


tokenizer = tiktoken.get_encoding('p50k_base')


# Get length of token from text
def tiktoken_len(text):
    tokens = tokenizer.encode(
        text,
        disallowed_special=()
    )
    return len(tokens)


# Function to get the text splitter object
def get_text_splitter():
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20,
        separators=["\n\n", "\n", " ", ""]
    )

    return text_splitter


# Function to get the embeddings object
def get_embeddings():
    embeddings = OpenAIEmbeddings(openai_api_key=openai_key,
                                  document_model_name=embedding_model_name, query_model_name=embedding_model_name)

    return embeddings


# Function to create a Pinecone index
def create_pinecone_index():
    pinecone.init(
        api_key=pinecone_api_key,
        environment=pinecone_env
    )

    pinecone.create_index(
        name=index_name,
        metric='dotproduct',
        dimension=1536
    )


# Function to perform similarity search on Pinecone index
def similarity_search(vectorstore, query):
    return vectorstore.similarity_search(query, k=3)


# Function to generate answers using LangChain
def generative_answering(chain, query):
    return chain.run(query)


# Function to upsert data to Pinecone index
def upsert_pinecone(record_data, text_splitter, embeddings, index):
    metadata = {
        'metadata': record_data['metadata']
    }

    # Split the record text into chunks
    print(record_data['content'])
    record_texts = text_splitter.split_text(record_data['content'])
    print(len(record_texts))

    # Create metadata dicts for each chunk
    record_metadata = [{
        "chunk": j, "text": text, **metadata
    } for j, text in enumerate(record_texts)]

    # Generate UUIDs for each chunk
    ids = [str(uuid4()) for _ in range(len(record_texts))]

    # Embed the chunks using SentenceTransformer
    embeds = embeddings.embed_documents(record_texts)

    # Upsert the chunks and metadata to Pinecone index
    for i in range(0, len(record_metadata), 80):
        index.upsert(vectors=zip(ids[i:i + 80], embeds[i:i + 80], record_metadata[i:i + 80]))


# Function to create a record object from text data
# noinspection PyBroadException
def make_record(text):
    chat = ChatOpenAI(openai_api_key=openai_key, temperature=0.5, model_name=openai_model_name)

    # Generate system message prompt for summarizing data
    system_template = """You are an excellent writer. The original data human gives is the Financial, Commercial and 
    Membership Information. You must rewrite data systematically so that document based conversational bot can 
    understand easily. The data must represent the whole contents of the original data. You mustn't generate the 
    original data. You must output rewritten data."""
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)

    # Generate human message prompt for raw data input
    human_template = '{raw_data}'
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    # Generate chat prompt from system and human message prompts
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
    chat_prompt = chat_prompt.format_prompt(raw_data=text).to_messages()

    # Use LangChain to summarize data
    try:
        summarized_data = chat(chat_prompt).content
        print(summarized_data)
    except:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        error_name = exc_type.__name__
        if error_name == 'RateLimitError' or error_name == 'InvalidRequestError':
            return_record = []
            for subrecord in text:
                print("+++++++++++++++++++++++++++++")
                summarized_data = subrecord.page_content
                metadata = subrecord.metadata['source']
                return_record.append({'metadata': metadata, 'content': summarized_data})
            return return_record
        else:
            summarized_data = "None"

    # Generate system message prompt for generating metadata
    system_template = """You are an excellent writer. The original data human gives is the Financial, Commercial and 
    Membership Information. You must generate metadata so that document based conversational bot can understand 
    easily. The metadata must be short but represent the whole basic properties of the original data. You mustn't 
    generate the original data. You must output summarized metadata."""
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)

    # Generate human message prompt for raw data input
    human_template = '{raw_data}'
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    # Generate chat prompt from system and human message prompts
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
    chat_prompt = chat_prompt.format_prompt(raw_data=text).to_messages()

    # Use LangChain to generate metadata
    metadata = chat(chat_prompt).content
    print(metadata)
    return [{'metadata': metadata, 'content': summarized_data}]


def make_record1(text):
    return [{'metadata': "the Financial, Commercial and Membership Information", 'content': text[0].page_content}]


# Function to upsert data from a file to Pinecone index
def upsert_file(filepath, ext):
    text_splitter = get_text_splitter()
    embeddings = get_embeddings()

    # Initialize Pinecone index
    pinecone.init(
        api_key=pinecone_api_key,
        environment=pinecone_env
    )
    index = pinecone.Index(index_name)

    # Load data from file based on file extension
    if ext == 'doc':
        loader_data = UnstructuredWordDocumentLoader(filepath)
        dt = loader_data.load()
        record_upsert = make_record(dt)
        for sub_record in record_upsert:
            upsert_pinecone(sub_record, text_splitter, embeddings, index)
    elif ext == 'docx':
        loader_data = UnstructuredWordDocumentLoader(filepath)
        dt = loader_data.load()
        record_upsert = make_record(dt)
        for sub_record in record_upsert:
            upsert_pinecone(sub_record, text_splitter, embeddings, index)
    elif ext == 'pdf':
        loader_data = PDFMinerLoader(filepath)
        dt = loader_data.load()
        record_upsert = make_record(dt)
        for sub_record in record_upsert:
            upsert_pinecone(sub_record, text_splitter, embeddings, index)
    elif ext == 'pptx':
        loader_data = UnstructuredPowerPointLoader(filepath)
        dt = loader_data.load()
        record_upsert = make_record(dt)
        for sub_record in record_upsert:
            upsert_pinecone(sub_record, text_splitter, embeddings, index)
    elif ext == 'ppt':
        loader_data = UnstructuredPowerPointLoader(filepath)
        dt = loader_data.load()
        record_upsert = make_record(dt)
        for sub_record in record_upsert:
            upsert_pinecone(sub_record, text_splitter, embeddings, index)
    elif ext == 'txt':
        loader_data = TextLoader(filepath, encoding='latin1')
        dt = loader_data.load()
        record_upsert = make_record(dt)
        for sub_record in record_upsert:
            upsert_pinecone(sub_record, text_splitter, embeddings, index)
    elif ext == 'csv':
        loader_data = CSVLoader(filepath, encoding='latin1')
        dt = loader_data.load()
        record_upsert = make_record(dt)
        for sub_record in record_upsert:
            upsert_pinecone(sub_record, text_splitter, embeddings, index)


def upsert_folder(folderpath, ext):
    text_splitter = get_text_splitter()
    embeddings = get_embeddings()

    # Initialize Pinecone index
    pinecone.init(
        api_key=pinecone_api_key,
        environment=pinecone_env
    )
    index = pinecone.Index(index_name)

    filenames_upsert = os.listdir(folderpath)

    for filename_upsert in filenames_upsert:
        filepath_upsert = os.path.join(folderpath, filename_upsert)
        print(filepath_upsert)
        # Load data from file based on file extension
        if ext == 'doc':
            loader_data = UnstructuredWordDocumentLoader(filepath_upsert)
            dt = loader_data.load()
            record_upsert = make_record(dt)
            for sub_record in record_upsert:
                upsert_pinecone(sub_record, text_splitter, embeddings, index)
        elif ext == 'docx':
            loader_data = UnstructuredWordDocumentLoader(filepath_upsert)
            dt = loader_data.load()
            record_upsert = make_record(dt)
            for sub_record in record_upsert:
                upsert_pinecone(sub_record, text_splitter, embeddings, index)
        elif ext == 'pdf':
            loader_data = PDFMinerLoader(filepath_upsert)
            dt = loader_data.load()
            record_upsert = make_record(dt)
            for sub_record in record_upsert:
                upsert_pinecone(sub_record, text_splitter, embeddings, index)
        elif ext == 'pptx':
            loader_data = UnstructuredPowerPointLoader(filepath_upsert)
            dt = loader_data.load()
            record_upsert = make_record(dt)
            for sub_record in record_upsert:
                upsert_pinecone(sub_record, text_splitter, embeddings, index)
        elif ext == 'ppt':
            loader_data = UnstructuredPowerPointLoader(filepath_upsert)
            dt = loader_data.load()
            record_upsert = make_record(dt)
            for sub_record in record_upsert:
                upsert_pinecone(sub_record, text_splitter, embeddings, index)
        elif ext == 'txt':
            loader_data = TextLoader(filepath_upsert, encoding='latin1')
            dt = loader_data.load()
            record_upsert = make_record1(dt)
            for sub_record in record_upsert:
                upsert_pinecone(sub_record, text_splitter, embeddings, index)
            os.remove(filepath_upsert)
        elif ext == 'csv':
            loader_data = CSVLoader(filepath_upsert, encoding='latin1')
            dt = loader_data.load()
            record_upsert = make_record(dt)
            for sub_record in record_upsert:
                upsert_pinecone(sub_record, text_splitter, embeddings, index)


if __name__ == '__main__':
    # csvpath = '.\\Database\\csv_files\\ALEX BRAND-PC2-reviewed and analyzed by VA_Sheet 1.csv'
    # loader = CSVLoader(csvpath, encoding='latin1')
    # data = loader.load()
    # record = make_record(data)
    # path = '.\\Database\\pdf_files\\Business Operations Checklist.pdf'
    # upsert_file(path, 'pdf')
    extension = 'txt'
    folder = '.\\Database\\txt_files'
    upsert_folder(folder, extension)
