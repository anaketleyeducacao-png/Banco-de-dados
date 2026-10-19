import os
from dotenv import load_dotenv
import psycopg2
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer

load_dotenv()

modelo = SentenceTransformer(
    "paraphrase-multilingual-MiniLM-L12-v2"
)

conn = psycopg2.connect(os.environ["DATABASE_URL"])
register_vector(conn)
cur = conn.cursor()

documentos = [
    ("1FN", "A primeira forma normal exige valores atômicos.", "semanas_1_6"),
    ("JOIN", "INNER JOIN combina linhas de tabelas relacionadas.", "semanas_1_6"),
    ("GROUP BY", "GROUP BY agrupa linhas para funções como COUNT e SUM.", "semanas_1_6"),
    ("Chave estrangeira", "Uma chave estrangeira referencia outra tabela.", "semanas_1_6"),
    ("Redis", "Redis é muito usado para cache e acesso chave-valor.", "extra"),
    ("NoSQL", "NoSQL inclui modelos documento, chave-valor, coluna e grafo.", "semanas_1_6"),
]

cur.execute("DELETE FROM documentos;")

for titulo, conteudo, fonte in documentos:
    embedding = modelo.encode(f"{titulo}. {conteudo}")

    cur.execute(
        '''
        INSERT INTO documentos
            (titulo, conteudo, fonte, embedding)
        VALUES (%s, %s, %s, %s)
        ''',
        (titulo, conteudo, fonte, embedding)
    )

conn.commit()

pergunta = "como fazer um JOIN no SQL"
query_embedding = modelo.encode(pergunta)

cur.execute(
    '''
    SELECT titulo, conteudo, fonte,
           1 - (embedding <=> %s) AS similaridade
    FROM documentos
    ORDER BY embedding <=> %s
    LIMIT 3
    ''',
    (query_embedding, query_embedding)
)

print("TOP-3 PARA:", pergunta)

for titulo, conteudo, fonte, similaridade in cur.fetchall():
    print(f"{similaridade:.4f} | {titulo} | {conteudo}")


cur.close()
conn.close()

