from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.text_splitter import CharacterTextSplitter
from vosk_tts import Model, Synth

def process_input(question):
    model_local = ChatOllama(model="solar:10.7b-instruct-v1-q5_0")
    
    # Convert string of URLs to list
    urls_list = [
    "https://www.centrinvest.ru/for-individuals/safe-boxes"
    
]
    docs = [WebBaseLoader(url).load() for url in urls_list]
    docs_list = [item for sublist in docs for item in sublist]
    
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=7500, chunk_overlap=100)
    doc_splits = text_splitter.split_documents(docs_list)

    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        collection_name="rag-chroma",
        embedding= OllamaEmbeddings(model='nomic-embed-text'),
    )
    retriever = vectorstore.as_retriever()

    after_rag_template = """Answer the question based only on the following context:
    {context}
    Question: {question}
    """
    after_rag_prompt = ChatPromptTemplate.from_template(after_rag_template)
    after_rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | after_rag_prompt
        | model_local
        | StrOutputParser()
    )
    return after_rag_chain.invoke(question)

def generate_speech(text, output_file, speaker_id=2):
    model = Model(model_name="vosk-model-tts-ru-0.6-multi")
    synth = Synth(model)
    synth.synth(text, output_file, speaker_id=speaker_id)

text = process_input("Привет! Как мне оформить аренду сейфа?")
print(text)


