from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import time
import pandas as pd
import sys
DATA_PATH = 'data/'
DB_FAISS_PATH = 'vectorstore/db_faiss'


# create vector database

def create_vector_db():
    start_time = time.time()
    loader = DirectoryLoader(DATA_PATH,
                             glob='*.pdf',
                             loader_cls=PyPDFLoader)
    documents = loader.load()
    end_time = time.time() - start_time
    print("1", end_time)
    print("documents", documents)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,
                                                   chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    end_time1 = time.time() - end_time
    print("2", end_time1)
    print("text_splitter", texts)
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                       model_kwargs={'device': 'cpu'})

    # print("embeddings", embeddings)
    # db = FAISS.from_documents(texts, embeddings)

    #############################################################################################

    db = FAISS.load_local(DB_FAISS_PATH, embeddings)
    # print("similarity", db.similarity_search_with_relevance_scores(query="amino acid", k=2))
    similar = db.similarity_search(query="amino_acid", k=2)
    print("similar", similar)
    # for score, index in similar:
    #     print(db.get_id(index))
    print("page_content:", similar[0].page_content)

    meta_data = similar[0].metadata
    print(meta_data)
    chunk_id = list(db.docstore._dict.keys())[0]
    print("chunk_id", chunk_id)
    df = store_to_df(db)
    print(df['content'])
    for i in range(len(df['content'])):
        if similar[0].page_content in df['content'][i]:
            print("got it", df['chunk_id'][i])
        # else:
        #     print("didn't")
    # sys.exit()
    ids = ['20ee8af7-0322-461f-8e30-41e410fbcbb9']
    # missing_ids = set(ids).difference(db.index_to_docstore_id.values())
    # print(set(ids))
    # print(db.index_to_docstore_id.values())
    # print("missing_ids", missing_ids)
    # print(db.delete(ids))
    # add_new_data = ['Amino acids are organic compounds that serve as the building blocks of proteins and play a crucial role in various biological processes. They are composed of carbon, hydrogen, oxygen, and nitrogen atoms, and some also contain sulfur. There are 20 different amino acids that can combine in different sequences to form a wide array of proteins.']
    # print("new_data_id",db.add_texts(texts=add_new_data, ids=['916cdce2-01ff-49a9-8002-7039c5ff171a'], metadatas=[meta_data]))
    # df = store_to_df(db)

    # print("chunk_id", df['chunk_id'][0])
    # Specify the file path where you want to save the Excel file
    excel_file_path = 'your_file.xlsx'

    # Save the DataFrame to Excel
    df.to_excel(excel_file_path, index=False)

    db.save_local(DB_FAISS_PATH)

def store_to_df(store):
    v_dict = store.docstore._dict
    data_rows = []
    for k in v_dict.keys():
        doc_name = v_dict[k].metadata['source'].split('/')[-1]
        page_number = v_dict[k].metadata['page'] + 1
        content = v_dict[k].page_content
        data_rows.append({"chunk_id": k, "document": doc_name, "page": page_number, "content": content})
    vector_df = pd.DataFrame(data_rows)
    return vector_df


if __name__ == '__main__':
    create_vector_db()
    print("something")
