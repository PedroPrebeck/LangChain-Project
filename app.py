import streamlit as st

def main():
  st.set_page_config(page_title="Chat with PDFs", page_icon=":books:")
  
  st.header("Chat with PDFs :books:")
  st.text_input("Ask any question about your documments:")
  
  with st.sidebar:
    st.subheader("Your documents")
    st.file_uploader("Upload your PDFs here :books:", accept_multiple_files=True)
    st.button("Upload")
    
if __name__ == '__main__':
  main()