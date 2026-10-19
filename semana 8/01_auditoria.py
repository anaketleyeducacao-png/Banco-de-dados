
# 01_auditoria.py

from dotenv import load_dotenv
from pathlib import Path
load_dotenv(Path(__file__).parent / ".env")
import os
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import os

engine = create_engine(os.environ["DATABASE_URL"])

df = pd.read_sql("""
    SELECT
        c.id AS cliente_id,
        c.cidade,
        c.email,
        COUNT(p.id) AS total_pedidos,
        ROUND(AVG(p.total)::numeric, 2) AS ticket_medio,
        SUM(p.total) AS total_gasto,
        MAX(CURRENT_DATE - p.criado_em::date) AS dias_inativo,
        COUNT(*) FILTER (WHERE p.status = 'cancelado') AS qtd_cancelados,
        COUNT(*) FILTER (WHERE p.status = 'entregue') AS qtd_entregues
    FROM clientes c
    LEFT JOIN pedidos p ON p.cliente_id = c.id
    GROUP BY c.id, c.cidade, c.email
""", engine)

print("=== 1. SHAPE ===")
print(df.shape)

print("\n=== 2. TIPOS ===")
print(df.dtypes)

print("\n=== 3. NULOS ===")
nulos = df.isnull().sum()
pct_nulos = (nulos / len(df) * 100).round(1)
print(pd.DataFrame({"nulos": nulos, "pct_%": pct_nulos}))

print("\n=== 4. DUPLICATAS ===")
print("Linhas duplicadas:", df.duplicated().sum())
print("IDs únicos:", df["cliente_id"].nunique(), "/", len(df))

print("\n=== 5. CIDADES ===")
print(df["cidade"].value_counts(dropna=False))

print("\n=== 6. OUTLIERS POR IQR ===")
for col in ["ticket_medio", "total_gasto", "dias_inativo"]:
    serie = df[col].dropna()
    Q1 = serie.quantile(0.25)
    Q3 = serie.quantile(0.75)
    IQR = Q3 - Q1
    qtd = ((serie < Q1 - 1.5 * IQR) | (serie > Q3 + 1.5 * IQR)).sum()
    print(f"{col}: {qtd}")
