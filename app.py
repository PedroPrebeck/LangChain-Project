import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.llms import HuggingFaceHub
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from streamlit_chat import message


def get_pdf_text(pdf_docs):
  text = ""
  for pdf in pdf_docs:
    reader = PdfReader(pdf)
    for page in reader.pages:
      text += page.extract_text()
  return text

def get_text_chunks(text):
  text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
  chunks = text_splitter.split_text(text)
  return chunks

def get_vector_store(text_chunks):
  embeddings = OpenAIEmbeddings()
  # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-base")
  vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
  return vectorstore

def get_conversation_chain(vectorstore):
  llm = ChatOpenAI()
  # llm = HuggingFaceHub(repo_id="OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5", model_kwargs={"temperature":0.5, "max_length":512})
  memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
  conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=vectorstore.as_retriever(), memory=memory)
  return conversation_chain

def handle_user_input(user_question):
  response = st.session_state.conversation({"question": user_question})
  st.session_state.chat_history = response["chat_history"]
  st.write(response)
    
  for i, message_str in enumerate(st.session_state.chat_history):
    if i % 2 ==0:
        content = message_str.content
        message(content, is_user=True, key=i)
    else:
        content = message_str.content
        message(content, key=i)

def main():
  load_dotenv()
  
  st.set_page_config(page_title="Chat with PDFs", page_icon=":books:")
  
  if "conversation" not in st.session_state:
    st.session_state.conversation = None
    
  if "chat_history" not in st.session_state:
    st.session_state.chat_history = None
  
  st.header("Chat with PDFs :books:")
  
  user_question = st.text_input("Ask any question about your documents:")
  if user_question:
    handle_user_input(user_question)
  
  with st.sidebar:
    st.subheader("Your documents")
    pdf_docs = st.file_uploader("Upload your PDFs here :books:", accept_multiple_files=True)
    if st.button("Upload"):
      with st.spinner("Processing your PDFs"):
        raw_text = get_pdf_text(pdf_docs)
        text_chunks = get_text_chunks(raw_text)      
        
        vectorstore = get_vector_store(text_chunks)
        
        st.session_state.conversation = get_conversation_chain(vectorstore)    
    
if __name__ == '__main__':
  main()