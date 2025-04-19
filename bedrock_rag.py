from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma.vectorstores import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import os
import time
from langchain_aws import ChatBedrock, BedrockEmbeddings

# Define paths for persistence
file_path = './WhoMovedMyCheese.pdf'
model_name = "anthropic.claude-3-5-haiku-20241022-v1:0"
persist_directory = f'./chroma_db/anthropic'

# Initialize embedding model
embed_model = BedrockEmbeddings(
    model_id=model_name)

# Initialize LLM
llm = ChatBedrock(
    model_id=model_name,
    model_kwargs=dict(temperature=0),
)


def create_and_persist_vector_store(file_path, embed_model, persist_directory):
    """Creates a Chroma vector store from a PDF and persists it to disk."""
    print("\nAlright, let's get this book processed!")
    time.sleep(0.5)
    print("\nFirst, we'll break down the text into manageable chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512, chunk_overlap=128, add_start_index=True)
    time.sleep(0.7)
    print("Text splitter configured. Now, loading the PDF...")
    loader = PyPDFLoader(file_path)
    time.sleep(0.8)
    print("PDF loaded! Now, splitting it into individual pages...")
    book_pages = loader.load_and_split(text_splitter)
    num_pages = len(book_pages)
    time.sleep(1)
    print(f"Phew! That's {num_pages} pages we've got. Quite a read!\n")
    time.sleep(0.6)
    print("Next up, creating a smart index of this book so I can quickly find information...")
    print("This involves understanding the meaning of each chunk - almost like teaching me to 'read' it.")
    time.sleep(1.2)
    print("Creating the vector store (think of it as the book's super-efficient index)...")
    vector_store = Chroma.from_documents(
        documents=book_pages, embedding=embed_model, persist_directory=persist_directory)
    time.sleep(1)
    print(
        f"\nBoom! The vector store is now created and saved safely in '{persist_directory}' for future use.")
    return vector_store


def load_vector_store(persist_directory, embed_model):
    """Loads an existing Chroma vector store from disk."""
    print(f"\nHey, I see you've processed this book before!")
    time.sleep(0.7)
    print(
        f"No need to do all that reading again. Let me quickly load the pre-built index from '{persist_directory}'...")
    vector_store = Chroma(persist_directory=persist_directory,
                          embedding_function=embed_model)
    time.sleep(1)
    print("Index loaded successfully! Ready to answer your questions.")
    return vector_store


def create_rag_chain(retriever, model):
    """Creates the Retrieval-Augmented Generation chain."""
    print("\nAlmost there! Now, let's set up the 'brain' that will look up information and answer your questions.")
    time.sleep(0.8)
    print("Configuring the question-answering system...")
    system_prompt = (
        """
      You are an assistant specialized in answering questions solely from the story you are provided.
      Use only the given context to form your responses.
      If the information is not provided or unclear, state explicitly that you don't know.
      Keep your answers brief.
      {context}
      """
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(
        model, prompt
    )
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    time.sleep(0.9)
    print("Question-answering system ready to go!\n")
    return rag_chain


if __name__ == "__main__":
    # Check if the vector store already exists
    if os.path.exists(persist_directory):
        vector_store = load_vector_store(persist_directory, embed_model)
    else:
        vector_store = create_and_persist_vector_store(
            file_path, embed_model, persist_directory)

    print("Creating the information retriever...")
    time.sleep(0.6)
    retriever = vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": 5})
    print("Retriever set up to find the most relevant parts of the book.\n")

    rag_chain = create_rag_chain(retriever, llm)

    user_input = ''
    print(
        f"Hey! I've now got the knowledge of '{os.path.basename(file_path)}' at my fingertips. What do you wanna know?\n")

    while True:
        user_input = input('You: ')
        if (user_input == 'Thank you'):
            print("\nYou're welcome! Let me know if you have any more questions later.")
            break
        print("Hmm, let me quickly check the book for that...")
        response = rag_chain.invoke(
            {"input": user_input})
        answer = response['answer']
        time.sleep(0.4)
        print(f'AI: {answer}')
        print('---'*30)
