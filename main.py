import streamlit as st
from PyPDF2 import PdfReader

from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatAnyscale
from langchain.document_loaders import TextLoader
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS

from htmlTemplates import css, user_template, bot_template

import os


def get_docs(docs):
    text = ""
    for doc in docs:
        loader = TextLoader(doc, encoding='utf-8')
        text += loader.load_and_split()
    print(f'{len(text)} file is saved')
    return text
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=20
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    model_name = "BAAI/bge-small-en"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    embeddings = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs)
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    condense_prompt = PromptTemplate.from_template(("""Given the following conversation and a follow up question,
            rephrase the follow up question to be a standalone question.

            Chat History:
            {chat_history}
            Follow Up Input: {question}
            Standalone question:"""))
    combine_docs_custom_prompt = PromptTemplate.from_template(("""Answer the question according to the given context.
             If the context does not have enough information to answer the question, 
             please respond with: "The context does not have enough information to answer the question.".

             {context}

             Question: {question}
             """))

    llm = ChatAnyscale(model_name="codellama/CodeLlama-34b-Instruct-hf",
                       temperature=0)

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        condense_question_prompt=condense_prompt,
        combine_docs_chain_kwargs=dict(prompt=combine_docs_custom_prompt)
    )
    return conversation_chain

def handle_userinput(user_question, db):
    docs = db.similarity_search(user_question)
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content),
                     unsafe_allow_html=True)
        else:
            msg = bot_template.replace("{{MSG}}", message.content)
            msg += f"REFERENCES: \n\n {docs[0].page_content}"
            st.write(msg,
                     unsafe_allow_html=True)

def main():

    st.set_page_config(page_title="Doc Summary📁")
    st.write(css, unsafe_allow_html=True)
    st.header("Doc Summary📁")

    try:
        print(os.environ["ANYSCALE_API_KEY"])
    except KeyError:
        from dotenv import load_dotenv

        load_dotenv()

        try:
            print(os.environ["ANYSCALE_API_KEY"])
        except KeyError:
            os.environ["ANYSCALE_API_KEY"] = st.secrets["ANYSCALE_API_KEY"]
        except Exception as e:
            raise Exception(f"OPENAI_API_KEY not found in environment variables or .env file: {e}")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "db" not in st.session_state:
        st.session_state.db = None

    with st.sidebar:
        st.subheader("Your documents")
        docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                raw_text = get_pdf_text(docs)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # create vector store
                vectorstore = get_vectorstore(text_chunks)
                st.session_state.db = vectorstore

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(
                    vectorstore)


    user_question = st.text_input("Ask a question about your documents:")
    if user_question and st.session_state.db:
        handle_userinput(user_question, st.session_state.db)




if __name__ == '__main__':
    main()

