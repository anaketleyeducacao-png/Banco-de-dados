
# 04_feature_engineering.py

from dotenv import load_dotenv
from pathlib import Path
load_dotenv(Path(__file__).parent / ".env")
import os
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

engine = create_engine(os.environ["DATABASE_URL"])

df = pd.read_sql("""
    SELECT
        c.id AS cliente_id,
        COALESCE(COUNT(p.id), 0) AS total_pedidos,
        COALESCE(SUM(p.total), 0) AS total_gasto,
        COALESCE(MAX(CURRENT_DATE - p.criado_em::date), 9999) AS dias_inativo,
        COUNT(*) FILTER (WHERE p.status = 'cancelado') AS qtd_cancelados
    FROM clientes c
    LEFT JOIN pedidos p ON p.cliente_id = c.id
    GROUP BY c.id
""", engine)

# A) taxa_cancelamento
df["taxa_cancelamento"] = (
    df["qtd_cancelados"] /
    df["total_pedidos"].replace(0, np.nan)
).fillna(0)

# B) cliente_ativo
df["cliente_ativo"] = (
    df["dias_inativo"] < 30
).astype(int)

# C) faixa_gasto
df["faixa_gasto"] = pd.cut(
    df["total_gasto"],
    bins=[-0.01, 500, 2000, 5000, np.inf],
    labels=["baixo", "medio", "alto", "vip"]
)

print("=== DISTRIBUIÇÃO faixa_gasto ===")
print(df["faixa_gasto"].value_counts(dropna=False))

print("\n=== DISTRIBUIÇÃO cliente_ativo ===")
print(df["cliente_ativo"].value_counts())

print("\n=== RESUMO taxa_cancelamento ===")
print(df["taxa_cancelamento"].describe())

print("\n=== AMOSTRA ===")
print(df[
    ["cliente_id", "taxa_cancelamento", "cliente_ativo", "faixa_gasto"]
].head(10))
