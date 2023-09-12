from langchain import ConversationChain
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chains import RetrievalQA
from langchain.prompts import (
    MessagesPlaceholder
)
from streamlit_chat import message

from utils import *
from config import *

# Set subheader for the Streamlit app
st.subheader("Chatbot with Langchain, ChatGPT, Pinecone, and Streamlit")

# Initialize session state variables for chat history
if 'responses' not in st.session_state:
    st.session_state['responses'] = ["How can I assist you?"]

if 'requests' not in st.session_state:
    st.session_state['requests'] = []

# Initialize ChatOpenAI model
llm = ChatOpenAI(openai_api_key=openai_key, model_name=openai_model_name)

# Initialize ConversationBufferWindowMemory object for conversation memory
if 'buffer_memory' not in st.session_state:
    st.session_state.buffer_memory = ConversationBufferWindowMemory(k=3, return_messages=True)

# Initialize SystemMessagePromptTemplate for chat prompt
system_msg_template = SystemMessagePromptTemplate.from_template(template="""
1. Answer the question as truthfully as possible using the provided contex. The more, the better
2. If the answer is contained within the text below, you must say the details deeply including what you think.
3. If the answer is not contained within the text below, you must say what you think about 
the user question in details.
""")

# Initialize HumanMessagePromptTemplate for chat prompt
human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")

# Initialize ChatPromptTemplate for chat prompt
prompt_template = ChatPromptTemplate.from_messages(
    [system_msg_template, MessagesPlaceholder(variable_name="history"), human_msg_template])

# Initialize ConversationChain object for conversation
conversation = ConversationChain(memory=st.session_state.buffer_memory, prompt=prompt_template, llm=llm, verbose=True)

# system_msg_template_gpt = SystemMessagePromptTemplate.from_template(template="""
# Answer to the query. But you have to do it systematically and logically.
# And
# """)
#
# # Initialize HumanMessagePromptTemplate for chat prompt
# human_msg_template_gpt = HumanMessagePromptTemplate.from_template(template="{input}")
#
# # Initialize ChatPromptTemplate for chat prompt
# prompt_template_gpt = ChatPromptTemplate.from_messages(
#     [system_msg_template_gpt, MessagesPlaceholder(variable_name="history"), human_msg_template_gpt])
#
# # Initialize ConversationChain object for conversation
# conversation_gpt = ConversationChain(memory=st.session_state.buffer_memory, prompt=prompt_template_gpt, llm=llm, verbose=True)

# Set title for the Streamlit app
st.title("Langchain Chatbot")

# Create container for chat history
response_container = st.container()

# Create container for text input
textcontainer = st.container()

embeddings = OpenAIEmbeddings(openai_api_key=openai_key,
                                  document_model_name=embedding_model_name, query_model_name=embedding_model_name)

[df_Amazon, df_investments, df_shopifyDTC] = Set_dataframe()

# Get user input from text input container
with textcontainer:
    query = st.text_input("Query: ", key="input")
    if query:
        with st.spinner("typing..."):
            # Initialize Pinecone index object
            index = pinecone.Index(index_name)

            conversation_string = get_conversation_string()
            next_query = query_refiner(conversation_string, query)
            st.subheader("Next Query:")
            st.write(next_query)

            # Create Vectorstore
            text_field = "text"
            vectorstore = Pinecone(
                index, embeddings.embed_query, text_field
            )
            chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vectorstore.as_retriever(
                    search_type="mmr",
                    search_kwargs={'k': 5},
                    return_source_documents=True
                )
            )
            context_langchain = chain.run(query)

            response = conversation.predict(input=f"Context:\n {context_langchain}\n"                                     
                                                  f"Query:\n{query}")
            print(response)
            # response_gpt = conversation_gpt.predict(input=f"Query:\n{query}")
            # response = response + "\n\n\nAlso" + response_gpt
        st.session_state.requests.append(query)
        st.session_state.responses.append(response)

# Display chat history in response container
with response_container:
    if st.session_state['responses']:
        for i in range(len(st.session_state['responses'])):
            message(st.session_state['responses'][i], key=str(i))
            if i < len(st.session_state['requests']):
                message(st.session_state["requests"][i], is_user=True, key=str(i) + '_user')
