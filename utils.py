from langchain.chat_models import ChatOpenAI
from sentence_transformers import SentenceTransformer
import pinecone
import openai
import streamlit as st
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
import pandas as pd
from pandasai import SmartDatalake
from pandasai.llm import OpenAI
from config import *

# Set OpenAI API key
openai.api_key = openai_key

# Initialize SentenceTransformer model
model = SentenceTransformer('all-mpnet-base-v2')

# Initialize Pinecone with API key and environment
pinecone.init(api_key=pinecone_api_key, environment=pinecone_env)


def find_match(input_query, index_):
    # Encode the input using SentenceTransformer model
    input_em = model.encode(input_query).tolist()

    # Perform similarity search on Pinecone index with the encoded input
    result = index_.query(input_em, top_k=3, includeMetadata=True)

    if len(result['matches']) < 3:
        return "Nothing Special"
    else:
        print("____________________")
        print(result['matches'][0]['metadata']['text'])
        print("____________________")
        print(result['matches'][1]['metadata']['text'])
        print("____________________")
        print(result['matches'][2]['metadata']['text'])
        print("____________________")
        return result['matches'][0]['metadata']['text'] + "\n" + result['matches'][1]['metadata']['text'] + "\n" + \
            result['matches'][2]['metadata']['text']


def query_refiner(conversation, query):
    # Use OpenAI's Completion API to refine the query based on conversation and user query
    prompt = """Given the following user query and conversation log, guess the next one question that would be the most 
    relevant to provide the user with an answer from a knowledge base. I want only one query. \n\nCONVERSATION 
    LOG is  \n""" + conversation + """\n\nQuery is """ + query + """\n\nNext Query:"""
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=openai_model_name,
        messages=messages,
        temperature=0.7,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response["choices"][0]["message"]["content"]


def get_conversation_string():
    conversation_string = ""
    for i in range(len(st.session_state['responses']) - 1):
        conversation_string += "Human: " + st.session_state['requests'][i] + "\n"
        conversation_string += "Bot: " + st.session_state['responses'][i + 1] + "\n"
    return conversation_string


def find_group(query):
    chat = ChatOpenAI(openai_api_key=openai_key, temperature=0.5, model_name=openai_model_name)
    # Generate system message prompt for summarizing data
    system_template = """You now have to determine which group the question is about in the input sentence.
    Group 1: MDS Amazon
    Group 2: MDS Investments
    Group 3: MDS ShopifyDTC
    !Important! You must answer just Group number like 1 or 2 or 3"""
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)

    # Generate human message prompt for raw data input
    human_template = '{query}'
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    # Generate chat prompt from system and human message prompts
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
    chat_prompt = chat_prompt.format_prompt(query=query).to_messages()
    group_num = chat(chat_prompt).content
    return group_num


def context_pandas(query, dataframe):
    llm = OpenAI(api_token=openai_key, model=openai_model_name)
    dl = SmartDatalake(dataframe, config={"llm": llm})
    answer = dl.chat(query)
    print(answer)
    chat = ChatOpenAI(openai_api_key=openai_key, temperature=0.5, model_name=openai_model_name)
    system_prompt = "If the answer is simple, please answer in a complete sentence that matches the question."

    # Generate human message prompt for raw data input
    human_template = "query: {query} \n\n answer: {answer}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages([system_prompt, human_message_prompt])
    chat_prompt = chat_prompt.format_prompt(query=query, answer=answer).to_messages()

    response = chat(chat_prompt).content
    print(response)

    return response


def Set_dataframe():
    df_amazon = [pd.read_csv("Database\\MDS\\MDS Amazon - Million Dollar Sellers Community_comments count analysis.csv"),
                 pd.read_csv("Database\\MDS\\MDS Amazon - Million Dollar Sellers Community_members analysis.csv"),
                 pd.read_csv("Database\\MDS\\MDS Amazon - Million Dollar Sellers Community_posts analysis.csv"),
                 pd.read_csv("Database\\MDS\\MDS Amazon - Million Dollar Sellers Community_posts count analysis.csv"),
                 pd.read_csv("Database\\MDS\\MDS Amazon - Million Dollar Sellers Community_Text DB.csv")]

    df_investments = [pd.read_csv("Database\\MDS\\MDS Investments - Million Dollar Sellers Community_commenters count analysis.csv"),
                      pd.read_csv("Database\\MDS\\MDS Investments - Million Dollar Sellers Community_members analysis.csv"),
                      pd.read_csv("Database\\MDS\\MDS Investments - Million Dollar Sellers Community_posts analysis.csv"),
                      pd.read_csv("Database\\MDS\\MDS Investments - Million Dollar Sellers Community_posters count analysis.csv"),
                      pd.read_csv("Database\\MDS\\MDS Investments - Million Dollar Sellers Community_Text DB.csv")]

    df_shopifydtc = [pd.read_csv("Database\\MDS\\MDS ShopifyDTC - Million Dollar Sellers Community_comments count analysis.csv"),
                     pd.read_csv("Database\\MDS\\MDS ShopifyDTC - Million Dollar Sellers Community_members analysis.csv"),
                     pd.read_csv("Database\\MDS\\MDS ShopifyDTC - Million Dollar Sellers Community_posts analysis.csv"),
                     pd.read_csv("Database\\MDS\\MDS ShopifyDTC - Million Dollar Sellers Community_posts counts analysis.csv"),
                     pd.read_csv("Database\\MDS\\MDS ShopifyDTC - Million Dollar Sellers Community_Text DB.csv")]

    return [df_amazon, df_investments, df_shopifydtc]


if __name__ == '__main__':
    query1 = "name of members who made more than Ivan Ong made post in MDS Amazon Group"
    [df1, df2, df3] = Set_dataframe()
    print(context_pandas(query1, df1))
