from dotenv import load_dotenv
import os
import streamlit as st

load_dotenv()

# create folder if not exists
os.makedirs("doc_files", exist_ok=True)


def process_doc(file_path):

    # document loading
    from langchain_community.document_loaders import PyPDFDirectoryLoader

    loader = PyPDFDirectoryLoader(file_path)
    docs = loader.load()

    # chunking
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(docs)

    # embedding model
    from langchain_community.embeddings import FastEmbedEmbeddings

    embedding = FastEmbedEmbeddings()

    # vector database
    from langchain_community.vectorstores import FAISS

    vector_db = FAISS.from_documents(chunks, embedding)

    # memory
    from langgraph.checkpoint.memory import InMemorySaver

    memory = InMemorySaver()

    # llm
    from langchain_groq import ChatGroq

    llm = ChatGroq(
        model="openai/gpt-oss-20b"
    )

    # tool
    from langchain_core.tools import tool

    @tool
    def retrieve_context(query: str):
        """
        Retrieve relevant chunks from vector database
        """

        context = ""

        con = vector_db.similarity_search(query, k=3)

        for i in con:
            context += i.page_content + "\n"

        return context

    # agent
    from langchain.agents import create_agent

    system_prompt = """
    You are an intelligent RAG assistant.

    Answer only from the retrieved context.

    If information is not present in context,
    say:
    'I could not find this information in uploaded documents.'
    """

    agent = create_agent(
        model=llm,
        tools=[retrieve_context],
        system_prompt=system_prompt,
        checkpointer=memory
    )

    st.session_state.agent = agent
    st.session_state.upload = True


# ---------------- SESSION STATE ---------------- #

if "upload" not in st.session_state:
    st.session_state.upload = False

if "agent" not in st.session_state:
    st.session_state.agent = None

if "messages" not in st.session_state:
    st.session_state.messages = []


# ---------------- UI ---------------- #

st.title("📄 Agentic RAG Chatbot")

# upload section
if not st.session_state.upload:

    up = st.file_uploader(
        label="Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True
    )

    if up:

        with st.spinner("Processing PDFs..."):

            path = "./doc_files"

            os.makedirs(path, exist_ok=True)

            for file in up:

                file_path = os.path.join(path, file.name)

                with open(file_path, "wb") as f:
                    f.write(file.getvalue())

            process_doc(path)

            st.success("Documents processed successfully")

            st.rerun()


# ---------------- CHAT UI ---------------- #

if st.session_state.upload and st.session_state.agent:

    # display old chats
    for m in st.session_state.messages:

        st.chat_message(
            m["role"]
        ).markdown(
            m["content"]
        )

    # user input
    query = st.chat_input("Ask anything from PDFs")

    if query:

        # display user msg
        st.chat_message("user").markdown(query)

        st.session_state.messages.append(
            {
                "role": "user",
                "content": query
            }
        )

        # agent response
        with st.spinner("Thinking..."):

            res = st.session_state.agent.invoke(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": query
                        }
                    ]
                },
                {
                    "configurable": {
                        "thread_id": "1"
                    }
                }
            )

            answer = res["messages"][-1].content

        # display assistant msg
        st.chat_message("assistant").markdown(answer)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )