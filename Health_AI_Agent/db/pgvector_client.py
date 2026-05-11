import psycopg2
from app.config import PG_HOST,PG_DB,PG_USER,PG_PASSWORD

conn=psycopg2.connect(
host=PG_HOST,
database=PG_DB,
user=PG_USER,
password=PG_PASSWORD
)

def similarity_search(q):

    cur=conn.cursor()

    cur.execute("SELECT content FROM documents LIMIT 5")

    rows=cur.fetchall()

    return [{"content":r[0]} for r in rows]
