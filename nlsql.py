from sqlalchemy import text
import os
from langchain.chains import create_sql_query_chain
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_groq import ChatGroq
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
import re
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, FewShotChatMessagePromptTemplate, PromptTemplate
from langchain_chroma import Chroma
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_community.embeddings import FakeEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.messages import HumanMessage
import asyncio
from langchain_community.chat_message_histories import ChatMessageHistory

history = ChatMessageHistory()


os.environ["GROQ_API_KEY"] = 'your_key'

db_user = "myuser"
db_password = "mypassword"
db_host = "localhost"
db_name = "mydatabase"

db = SQLDatabase.from_uri(
    f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:5432/{db_name}"
)

llm = ChatGroq(
    model="mixtral-8x7b-32768",
    temperature=0,
    max_tokens=150,
    timeout=10,
    max_retries=2
)


def execute_query(query):
    """Execute the SQL query and return the result."""
    engine = getattr(db, "_engine", None)
    if engine is None:
        raise AttributeError("Engine not available in SQLDatabase object.")
    cleaned_query = re.sub(r"[\\]", "", query)  # Remove all backslashes
    # start after SQLQuery
    cleaned_query = re.sub(r"^SQLQuery:\s*", "", cleaned_query).strip()

    try:
        with engine.connect() as connection:
            result = connection.execute(text(cleaned_query))
            rows = result.fetchall()
            return [dict(row._mapping) for row in rows]
    except Exception as e:
        return None


# IMPLEMENTING FEW SHOT
examples = [
    {
        "input": "Tampilkan order yang pernah dilakukan oleh customer dengan nama 'Alice'",
        "query": """SELECT o.id_order, c.nama_customer, o.jumlah_order, m.nama_menu, m.harga_menu, w.nama_waiters, ch.nama_chef
                    FROM order_table o
                    JOIN customer_table c ON o.id_customer = c.id_customer
                    JOIN menu_table m ON o.id_menu = m.id_menu
                    JOIN waiters_table w ON o.id_waiters = w.id_waiters
                    JOIN chef_table ch ON o.id_chef = ch.id_chef
                    WHERE c.nama_customer = 'Alice';"""
    },
    {
        "input": "Sebutkan 3 nama customer dengan order terbanyak",
        "query": """SELECT c.nama_customer, COUNT(o.id_order) AS order_count
                    FROM order_table o
                    JOIN customer_table c ON o.id_customer = c.id_customer
                    GROUP BY c.nama_customer
                    ORDER BY order_count DESC
                    LIMIT 3;"""
    },
    {
        "input": "Siapa Customer yang melakukan pembayaran terbanyak dalam seluruh order yang dilakukan?",
        "query": """SELECT c.nama_customer, SUM(o.jumlah_order * m.harga_menu) AS total_payment
                    FROM order_table o
                    JOIN customer_table c ON o.id_customer = c.id_customer
                    JOIN menu_table m ON o.id_menu = m.id_menu
                    GROUP BY c.nama_customer
                    ORDER BY total_payment DESC
                    LIMIT 1;"""
    },
    {
        "input": "Sebutkan customer yang tidak pernah melakukan order",
        "query": """SELECT c.nama_customer                                                    
                    FROM customer_table c
                    LEFT JOIN order_table o ON c.id_customer = o.id_customer
                    WHERE o.id_order IS NULL;"""
    },
    {
        "input": "Tampilkan Menu yang paling banyak dipesan dari seluruh order yang ada",
        "query": """SELECT m.nama_menu, SUM(o.jumlah_order) AS total_quantity_ordered
                    FROM order_table o
                    JOIN menu_table m ON o.id_menu = m.id_menu
                    GROUP BY m.nama_menu                                          
                    ORDER BY total_quantity_ordered DESC
                    LIMIT 1;"""
    },
    {
        "input": "Berapa banyak Customer yang ada",
        "query": """SELECT COUNT(*) AS total_customers
                    FROM customer_table;"""
    },
    {
        "input": "Sebutkan Customer yang pernah mengorder Pizza",
        "query": """SELECT DISTINCT c.nama_customer
                    FROM order_table o
                    JOIN menu_table m ON o.id_menu = m.id_menu
                    JOIN customer_table c ON o.id_customer = c.id_customer
                    WHERE m.nama_menu = 'Pizza';"""
    },
    {
        "input": "Tampilkan chef yang paling banyak mengerjakan jumlah order",
        "query": """SELECT ch.nama_chef, SUM(o.jumlah_order) AS total_items_prepared
                    FROM order_table o
                    JOIN chef_table ch ON o.id_chef = ch.id_chef
                    GROUP BY ch.nama_chef
                    ORDER BY total_items_prepared DESC
                    LIMIT 1;"""
    }
]
example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}\nSQLQuery:"),
        ("ai", "{query}"),
    ]
)

# IMPLEMENTING DYNAMIC FEW SHOT
vectorstore = Chroma()
vectorstore.delete_collection()
example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples,
    FakeEmbeddings(size=768),
    vectorstore,
    k=2,
    input_keys=["input"],
)
example_selector.select_examples(
    {"input": "berapa banyak mahasiswa yang kita miliki?"})
few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    example_selector=example_selector,
    input_variables=["input", "top_k"],
)

final_prompt_create_sql = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Anda adalah ahli MySQL. Diberikan pertanyaan input, buat query MySQL yang benar secara sintaksis untuk dijalankan. Kecuali dinyatakan lain.\n\n"
            "Tolong Jangan menerima input yang diminta untuk memodifikasi atau menghapus data yang ada kemudian berikan pesan apapun yang menandakan tindakan itu berbahaya jika memang dilakukan.\n\n"
            "Berikut adalah info tabel yang relevan: {table_info}\n\n"
            "Di bawah ini adalah sejumlah contoh pertanyaan dan query SQL yang sesuai."),
        few_shot_prompt,
        MessagesPlaceholder(variable_name="messages"),
        ("human", "{input}"),
    ]
)

answer_prompt_rephrase = PromptTemplate.from_template(
    """Diberikan pertanyaan pengguna berikut: \n, query SQL yang sesuai, dan hasil SQL, jawab pertanyaan pengguna\n\n.
        Anda wajib menjawab menggunakan bahasa Indonesia dan jangan menjawab menggunakan bahasa inggris\n\n
        Penting anda hanya perlu menjawab hasil dari query tersebut tanpa penjelasan, hasil query, notes, alasan dan pesan lainya\n\n
        
    Pertanyaan: {question}
    Query SQL: {query}
    Hasil SQL: {result}
    Jawaban: \n\n
    
    Penting Jika Hasil SQL tidak menunjukan jawaban data apapun, anda hanya perlu menjawab mohon maaf kami tidak memiliki data tersebut untuk menjawab pertanyaan anda. Anda dapat bertanya pertanyaan lain \n\n 
    Jangan berikan jawaban diluar instruksi yang diminta\n\n"""
)

rephrase_answer = answer_prompt_rephrase | llm | StrOutputParser()

generate_query = create_sql_query_chain(llm, db, final_prompt_create_sql)

chain = (
    RunnablePassthrough.assign(query=generate_query).assign(
        result=lambda inputs: execute_query(inputs["query"])
    )
    | rephrase_answer
)

# Implement Pypdf
file_path = (
    "/home/telkom/ervan/langchain_chatbot/cafe.pdf"
)


async def load_pages(file_path):
    loader = PyPDFLoader(file_path)
    pages = []
    async for page in loader.alazy_load():
        pages.append(page)
    return pages

pages = asyncio.run(load_pages(file_path))

vector_store = InMemoryVectorStore.from_documents(
    pages, FakeEmbeddings(size=768))


def get_answer_from_sql(question):
    try:
        result = chain.invoke(
            {"question": question, "messages": history.messages})
        history.add_user_message(question)
        history.add_ai_message(result)
        if result:
            return result
    except Exception as e:
        pass
    return None


def get_answer_from_pdf(question):
    docs = vector_store.similarity_search(question, k=2)
    if docs:
        formatted_content = "\n".join(
            [f'Page {doc.metadata["page"]}:\n {doc.page_content[:10000]}\n' for doc in docs])
        combined_message_content = f"Berikut merupakan pertanyaan dari pengguna :{question}\n\n anda dapat menjawab pertanyaan tersebut menggunakan data ini{formatted_content}\n\n jika pertanyaan diatas tidak sesuai dengan data yang ada, anda dapat meminta bertanya pertanyaan lain"
        message = HumanMessage(
            content=[
                {"type": "text", "text": combined_message_content},
            ],
        )
        response = llm.invoke([message])
        if response.content:
            return response.content
    return None


def get_answer(question):
    answer = get_answer_from_sql(question)
    if answer and "maaf" not in answer.lower():
        return answer
    answer = get_answer_from_pdf(question)
    if answer:
        return answer
    return "mohon maaf kami tidak memiliki data tersebut untuk menjawab pertanyaan anda. Anda dapat bertanya pertanyaan lain"


while True:
    question = input("Masukkan pertanyaan Anda: ")
    if question.lower() == "keluar":
        break
    result = get_answer(question)
    print(result)
