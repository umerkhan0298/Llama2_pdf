from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
import chainlit as cl

DB_FAISS_PATH = 'vectorstore/db_faiss'
# Use the following given information to answer the user's question. Make sure you answer only related to given information.
# If you don't know the answer, just say that I don't know, don't try to make up an answer.
custom_prompt_template = """Please base your response solely on the provided information and answer the user's question accordingly.
 If you don't have information on the topic, please respond with 'I don't know,' and avoid speculating or creating information.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""


# Loading the model
def load_llm():
    # Load the locally downloaded model here
    llm = CTransformers(
        model="TheBloke/Llama-2-7B-Chat-GGML",
        model_type="llama",
        max_new_tokens=512,
        temperature=0.5
    )
    return llm


def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    return prompt


# Retrieval QA Chain
def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                           chain_type='stuff',
                                           retriever=db.as_retriever(search_kwargs={'k': 2}),
                                           return_source_documents=False,
                                           chain_type_kwargs={'prompt': prompt}
                                           )
    return qa_chain


# QA Model Function
def qa_bot():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings)
    print("db", db)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)

    return qa


# output function
def final_result(query):
    qa_result = qa_bot()
    response = qa_result({'query': query})
    return response


qa_result = qa_bot()
while True:
    query = input("ask anything related to pdf\n")
    print(qa_result)
    if query == 'q':
        break
    # response = qa_result(query, return_only_outputs=True)
    response = qa_result({'query': query})
    print(response)


# print(final_result("who are you"))
# # chainlit code
# @cl.on_chat_start
# async def start():
#     chain = qa_bot()
#     msg = cl.Message(content="Starting the bot...")
#     await msg.send()
#     msg.content = "Hi, Welcome to Banking Bot. What is your query?"
#     await msg.update()
#
#     cl.user_session.set("chain", chain)
#
#
# @cl.on_message
# async def main(message):
#     chain = cl.user_session.get("chain")
#     cb = cl.AsyncLangchainCallbackHandler(
#         stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
#     )
#     # cb.answer_reached = True
#     res = await chain.acall(message, callbacks=[cb])
#     print(res)
#     answer = res["result"]
#     # sources = res["source_documents"]
#     #
#     # if sources:
#     #     answer += f"\nSources:" + str(sources)
#     # else:
#     #     answer += "\nNo sources found"
#
#     await cl.Message(content=answer).send()
