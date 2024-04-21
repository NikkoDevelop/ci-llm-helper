from llama_index.
from pathlib import Path
import chromadb
from llama_index import VectorStoreIndex, ServiceContext, download_loader
from llama_index.storage.storage_context import StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore 


SimpleCSVReader = download_loader("SimpleCSVReader")
loader = SimpleCSVReader(encoding="utf-8")
documents = loader.load_data(file=Path('./fine_food_reviews_1k.csv'))

client = chromadb.PersistentClient(path="./chroma_db_data")
chroma_collection = client.create_collection(name="reviews")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

llm = Ollama(model="mixtral")
service_context = ServiceContext.from_defaults(llm=llm, embed_model="local")

index = VectorStoreIndex.from_documents(documents, service_context=service_context, storage_context=storage_context)
query_engine = index.as_query_engine()

response = query_engine.query("What are the thoughts on food quality?")
print(response)