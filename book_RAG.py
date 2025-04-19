import os
from typing import List
import time

from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaEmbeddings

from langchain_core.runnables import Runnable
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain


from logger.logger import get_logger
from utils.formatter import get_formatted_time
import config


logger = get_logger(log_file_path="./bookRAG.log")


def load_pdf(pdf_path: str) -> List[Document]:
    """Loads documents from a PDF file."""
    print("\nAlright, let's get this book processed!")
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found at {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    file_name = pdf_path.rsplit('/')[-1]
    print(f"Loaded {len(docs)} pages from {file_name}")
    return docs


def split_documents(docs: List[Document], chunk_size: int, chunk_overlap: int) -> List[Document]:
    """Splits loaded documents into smaller chunks."""
    print("\nFirst, we'll break down the text into manageable chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap, add_start_index=True
    )
    print("Now, splitting it into individual pages...")
    splits = text_splitter.split_documents(docs)
    print(f"Phew! That's {len(splits)} chunks we've got. Quite a read!\n")
    return splits


def get_embeddings_model(model_name: str) -> OllamaEmbeddings:
    """Initializes and returns the Ollama embeddings model."""
    print(f"Initializing embeddings model: {model_name}")

    if model_name.startswith('huggingface'):
        return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Assumes Ollama is running and serving the specified model for embeddings
    return OllamaEmbeddings(model=model_name)


def create_and_persist_vector_store(file_path, embeddings, persist_directory):
    """Creates a Chroma vector store from a PDF and persists it to disk."""
    start_time = time.perf_counter()
    docs = load_pdf(file_path)
    splits = split_documents(docs, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
    end_time = time.perf_counter()

    logger.info('Time taken to load and split files: %s',
                get_formatted_time(start_time, end_time))
    print(
        "Next up, creating a smart index of this book so I can quickly find information..."
    )
    print(
        "This involves understanding the meaning of each chunk - almost like teaching me to 'read' it."
    )
    print(
        "Creating the vector store (think of it as the book's super-efficient index)..."
    )
    logger.info(
        'Creating vector store for %s using model: %s', file_path, config.EMBEDDING_MODEL_NAME)
    start_time = time.perf_counter()
    vector_store = Chroma.from_documents(
        documents=splits, embedding=embeddings, persist_directory=persist_directory
    )
    end_time = time.perf_counter()

    print('Time taken to create store: %s',
          get_formatted_time(start_time, end_time))
    logger.info('Time taken to create store: %s',
                get_formatted_time(start_time, end_time))

    return vector_store


def setup_vector_store(file_path, embeddings: HuggingFaceEmbeddings, store_path: str) -> Chroma:
    """Creates or loads a Chroma vector store."""

    if os.path.exists(store_path):
        print(f"Loading existing vector store from: {store_path}")
        vectorstore = Chroma(
            persist_directory=store_path,
            embedding_function=embeddings
        )
        print("Index loaded successfully! Ready to answer your questions.")
    else:
        logger.info(
            '-----------------------------------------------------------------------')
        logger.info(
            'Processing %s using model: %s', file_path, config.EMBEDDING_MODEL_NAME)
        logger.info(
            '-----------------------------------------------------------------------')
        vectorstore = create_and_persist_vector_store(
            file_path=file_path, embeddings=embeddings, persist_directory=store_path)
        print(
            f"\nBoom! The vector store is now created and saved safely in '{store_path}' for future use."
        )
    return vectorstore


def get_llm(model_name: str) -> ChatOllama:
    """Initializes and returns the Ollama chat model."""
    print(f"Initializing LLM: {model_name}")
    # Assumes Ollama is running and serving the specified chat model
    return ChatOllama(model=model_name, temperature=0)


def create_history_aware_retriever_chain(llm: ChatOllama, retriever: Chroma) -> Runnable:
    """Creates a chain to generate search queries based on chat history."""
    contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    return create_history_aware_retriever(llm, retriever, contextualize_q_prompt)


def create_qa_chain(llm: ChatOllama) -> Runnable:
    """Creates a chain to answer questions based on retrieved context."""
    qa_system_prompt = """
You are an expert Question Answering assistant. You have been provided with the complete text of a book. Your task is to answer the user's question based *exclusively* on the information contained within this book text.

**Instructions:**

1.  **Read the entire provided book context carefully.**
2.  **Analyze the user's question.**
3.  **Locate the relevant passages or information within the book text that directly address the question.**
4.  **Synthesize an answer using *only* the information found in the book.** Do not infer information not explicitly stated or implicitly strongly supported by the text.
5.  **Do not use any external knowledge, prior training data, or information outside of the provided book text.** Your knowledge is strictly limited to this document.
6.  **If the answer cannot be found within the book text, state clearly that the book does not contain the information needed to answer the question.** Do not attempt to guess or fabricate an answer.
7.  **Cite specific parts of the text to support your answer whenever possible (optional, but helpful).**
8.  **Keep the answers short and precise**
**[BEGIN BOOK TEXT]**

---
{context}
---

**[END BOOK TEXT]**

"""

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    return create_stuff_documents_chain(llm, qa_prompt)


def create_conversational_rag_chain(history_aware_retriever_chain: Runnable, qa_chain: Runnable) -> Runnable:
    """Creates the full conversational RAG chain."""
    print("Creating the full RAG chain...")
    return create_retrieval_chain(history_aware_retriever_chain, qa_chain)


def run_chat_loop(rag_chain: Runnable):
    """Runs the interactive chat loop."""
    chat_history: List[BaseMessage] = []
    end_phrase = ['bye', 'quit']
    print(f"\n--- Ready to Chat (type {'/'.join(end_phrase)} to exit) ---")
    while True:
        try:
            user_input = input("You: ")
            if user_input.strip() == '':
                continue
            if user_input.lower() in end_phrase:
                break

            processing_start = time.perf_counter()
            # Invoke the RAG chain
            response = rag_chain.invoke(
                {"input": user_input, "chat_history": chat_history})

            # Print the answer
            ai_response = response.get(
                'answer', 'Sorry, I could not generate a response.')
            print(f"AI: {ai_response}")
            processing_end = time.perf_counter()

            print('Time taken: %s' % get_formatted_time(
                processing_start, processing_end))

            # Update chat history
            chat_history.append(HumanMessage(content=user_input))
            chat_history.append(AIMessage(content=ai_response))

        except Exception as e:
            print(f"An error occurred: {e}")

    print("--- Chat Ended ---")


def main():
    """Main function to set up and run the RAG pipeline."""
    try:
        file_name = config.PDF_PATH.rsplit('/', maxsplit=1)[-1]
        vector_store_path = f'{config.VECTOR_STORE_ROOT}/{config.EMBEDDING_MODEL_NAME}/{file_name}'

        embeddings = get_embeddings_model(config.EMBEDDING_MODEL_NAME)

        vectorstore = setup_vector_store(
            config.PDF_PATH, embeddings, vector_store_path)
        retriever = vectorstore.as_retriever()

        llm = get_llm(config.CHAT_MODEL_NAME)

        history_retriever_chain = create_history_aware_retriever_chain(
            llm, retriever)
        qa_chain = create_qa_chain(llm)
        rag_chain = create_conversational_rag_chain(
            history_retriever_chain, qa_chain)

        run_chat_loop(rag_chain)

    except FileNotFoundError as e:
        print(f"Error: {e}. Please check the PDF_PATH configuration.")
    except ImportError as e:
        print(
            f"Import Error: {e}. Make sure all required libraries are installed.")
    except ValueError as e:
        print(f"Value Error: {e}. Check the values being used.")
    except OSError as e:
        print(f"OS Error: {e}. There was a problem with a file operation.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()
