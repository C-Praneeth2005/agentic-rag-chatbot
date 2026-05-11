from dotenv import load_dotenv
load_dotenv()


def process_doc(file_path):
    # doc loading
    from langchain_community.document_loaders import PyPDFLoader, PyPDFDirectoryLoader
    loader = PyPDFDirectoryLoader(file_path)
    docs = loader.load()

    # chunking
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    # embedding model
    from langchain_community.embeddings import FastEmbedEmbeddings
    embedding = FastEmbedEmbeddings()

    # indexing
    from langchain_community.vectorstores import FAISS
    vector_db = FAISS.from_documents(chunks, embedding)
    # agent = tool,llm,sysprompt,memory

    from langgraph.checkpoint.memory import InMemorySaver
    memory = InMemorySaver()
    from langchain.agents import create_agent
    from langchain_core.tools import tool
    from langchain_groq import ChatGroq
    llm = ChatGroq(model="openai/gpt-oss-20b")

    @tool
    def retrieve_context(query):
        """
        Retrieve the chunks relevant to the given query from the knowledge base

        """
        context = ""
        con = vector_db.similarity_search(query, k=3)
        for i in con:
            context = context + i.page_content + "\n"
        return context

    system_prompt = """
    you are a intelligent assistant and you can answer from the retrieved context
    """
    agent = create_agent(model=llm,
                         tools=[retrieve_context],
                         system_prompt=system_prompt,
                         checkpointer=memory
                         )
    st.session_state.agent = agent
    st.session_state.upload = True


# UI
import streamlit as st

if "upload" not in st.session_state:
    st.session_state.upload = False

if "agent" not in st.session_state:
    st.session_state.agent = None

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "messages" not in st.session_state:
    st.session_state.messages = []


# upload ui
if not st.session_state.upload:
    up = st.file_uploader(label="Upload a PDF", type=["pdf"], accept_multiple_files=True)
    if up:
        with st.spinner("Processing PDF..."):
            path = "./doc_files/"
            for file in up:
                with open(path + file.name, 'wb') as f:
                    f.write(file.getvalue())
            process_doc(path)
            st.rerun()

#chatui
if st.session_state.upload and st.session_state.agent :
    query=st.chat_input("ask anything")
    if query:
        st.session_state.messages.append({"role":"user","content":query})
        res=st.session_state.agent.invoke({"messages":[{"role":"user","content":query}]},
                                   {"configurable": {"thread_id": "1"}})
        st.session_state.messages.append({"role":"AI","content":res["messages"][-1].content})
        for m in st.session_state.messages:
            role=m["role"]
            content=m["content"]
            st.chat_message(role).markdown(content)




