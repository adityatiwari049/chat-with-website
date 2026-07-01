import os
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpointEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Some LangChain versions moved these into langchain_classic — try both.
try:
    from langchain.chains import create_history_aware_retriever, create_retrieval_chain
except ImportError:
    from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain

try:
    from langchain.chains.combine_documents import create_stuff_documents_chain
except ImportError:
    from langchain_classic.chains.combine_documents import create_stuff_documents_chain


load_dotenv()


# LLM setup

llm_endpoint = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen3-4B-Instruct-2507",
    huggingfacehub_api_token=os.getenv("HF_TOKEN"),
)
llm = ChatHuggingFace(llm=llm_endpoint)


# Embedding model setup


embeddings = HuggingFaceEndpointEmbeddings(
    model="BAAI/bge-small-en-v1.5",
    huggingfacehub_api_token=os.getenv("HF_TOKEN"),
)



# Step 1: Build a vector store from a website URL

def get_vectorstore_from_url(url: str) -> Chroma:
    """Scrape a website, split it into chunks, and embed it into Chroma."""
    loader = WebBaseLoader(url)
    document = loader.load()

    text_splitter = RecursiveCharacterTextSplitter()
    document_chunks = text_splitter.split_documents(document)

    vector_store = Chroma(collection_name = "website",
                          embedding_function=embeddings,
                           persist_directory="./chroma_db")

    # Clear out any leftover data from a previously-indexed website before
    # adding new content — otherwise old and new sites mix together and
    # retrieval returns stale/irrelevant answers.
    existing = vector_store.get()
    if existing["ids"]:
        vector_store.delete(ids=existing["ids"])

    vector_store.add_documents(document_chunks)

    return vector_store



# Step 2: History-aware retriever

def get_context_retriever_chain(vector_store: Chroma):
    retriever = vector_store.as_retriever()

    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up "
                  "in order to get information relevant to the conversation"),
    ])

    return create_history_aware_retriever(llm, retriever, prompt)



# Step 3: Conversational RAG chain

def get_conversational_rag_chain(retriever_chain):
    answer_prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's questions based on the below context. "
           "Give a complete, well-rounded answer that fully addresses the question. "
           "If the question requires a detailed explanation, you may go into depth but if it does not says depht give answer in medium lengght , "
           "but keep your response under 600 words which fully addrress the query . "
           "Avoid unnecessary rambling or repeating the context verbatim.\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])

    document_chain = create_stuff_documents_chain(llm, answer_prompt)
    return create_retrieval_chain(retriever_chain, document_chain)



# Step 4: Generate a response for the current user query

def get_response(user_input: str) -> str:
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)

    result = conversation_rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_input,
    })
    return result["answer"]


# Streamlit app

st.set_page_config(page_title="Chat with Website", page_icon="🤖")
st.title("Chat with website")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello, I am a bot. How can I help you?"),
    ]

with st.sidebar:
    st.header("Settings")
    website_url = st.text_input("Website URL")

if website_url is None or website_url == "":
    st.info("Please enter a website url")
else:
    # Build the vector store only once per URL — avoids re-scraping and
    # re-embedding the site on every Streamlit rerun.
    if "vector_store" not in st.session_state or st.session_state.get("last_url") != website_url:
        with st.spinner("Loading and indexing website..."):
            st.session_state.vector_store = get_vectorstore_from_url(website_url)
            st.session_state.last_url = website_url

    user_query = st.chat_input("Type your message here........")
    if user_query is not None and user_query != "":
        response = get_response(user_query)
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=response))

    # Render full conversation
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("human"):
                st.write(message.content)