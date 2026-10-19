
# 00_testar_ambiente.py
# Execute este arquivo primeiro.

from dotenv import load_dotenv
from pathlib import Path
load_dotenv(Path(__file__).parent / ".env")
import os
import pandas as pd
import sklearn
import sqlalchemy
import pyarrow
from sqlalchemy import create_engine


url = os.getenv("DATABASE_URL")

if not url:
    raise RuntimeError(
        "DATABASE_URL não encontrada. "
        "No PowerShell, configure com: "
        "$env:DATABASE_URL='postgresql://neondb_owner:npg_PLImbx4OC2tl@ep-tiny-art-b6x5ceao-pooler.c-2.sa-east-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require'"
    )

print("pandas:", pd.__version__)
print("scikit-learn:", sklearn.__version__)
print("SQLAlchemy:", sqlalchemy.__version__)

engine = create_engine(url)

# Verifica a conexão e as tabelas usadas na prática.
with engine.connect() as conn:
    tabelas = pd.read_sql("""
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'public'
          AND table_name IN ('clientes', 'pedidos')
        ORDER BY table_name
    """, conn)

print("\nTabelas encontradas:")
print(tabelas.to_string(index=False))

print("\nAmostra de clientes:")
print(pd.read_sql("SELECT * FROM clientes LIMIT 3", engine))

print("\nAmostra de pedidos:")
print(pd.read_sql("SELECT * FROM pedidos LIMIT 3", engine))

print("\nAMBIENTE OK")
