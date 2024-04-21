def rag_answer_from_text(input_text, question):
    from langchain_community.vectorstores import Chroma
    from langchain_community.embeddings import OllamaEmbeddings
    from langchain_community.chat_models import ChatOllama
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain.text_splitter import CharacterTextSplitter
    
    model_local = ChatOllama(model="solar:10.7b-instruct-v1-q5_0")
    
    # Split text into chunks
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=7500, chunk_overlap=100)
    doc_splits = text_splitter.split_text(input_text)
    
    # Convert text chunks to Embeddings and store them
    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        collection_name="rag-chroma",
        embedding=OllamaEmbeddings(model='nomic-embed-text'),
    )
    retriever = vectorstore.as_retriever()
    
    # Use RAG to answer the question
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

# Пример использования:

input_text = """
Как арендовать
Для заключения договора аренды индивидуальной сейфовой ячейки необходимо:
1. Предъявить документ, удостоверяющий личность.
2. Внести арендную плату.
3. Внести залоговую стоимость замка и ключа. Залоговая стоимость замка и ключа возвращается арендатору по окончании договора аренды, при условии возврата арендатором ключа.
Вы можете предоставить право пользования сейфовой ячейкой третьему лицу на основании нотариально оформленной доверенности. Кроме того, можно оформить договор аренды на двух арендаторов. Это особенно удобно для осуществления расчетов по сделкам купли-продажи недвижимости.

К услугам арендаторов индивидуальных сейфовых ячеек — оборудование для пересчета и проверки подлинности купюр, находящееся в изолированных кабинетах.

Информационно-справочная служба: (863) 2-000-000.
"""
question = "Как арендовать"
rag_output = rag_answer_from_text(input_text, question)
print(rag_output)
