
# 03_standard_scaler.py
# O exercício 3 é independente: ele extrai as features e faz split + scaling.
from dotenv import load_dotenv
from pathlib import Path
load_dotenv(Path(__file__).parent / ".env")
import os
import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

engine = create_engine(os.environ["DATABASE_URL"])

df = pd.read_sql("""
    SELECT
        c.id AS cliente_id,
        COUNT(p.id) AS total_pedidos,
        COALESCE(ROUND(AVG(p.total)::numeric, 2), 0) AS ticket_medio,
        COALESCE(SUM(p.total), 0) AS total_gasto,
        COALESCE(MAX(CURRENT_DATE - p.criado_em::date), 9999) AS dias_inativo
    FROM clientes c
    LEFT JOIN pedidos p ON p.cliente_id = c.id
    GROUP BY c.id
""", engine)

cols = [
    "total_pedidos",
    "ticket_medio",
    "total_gasto",
    "dias_inativo"
]

X = df[cols].astype(float)

# IMPORTANTE: dividir ANTES de ajustar o scaler.
X_train, X_test = train_test_split(
    X,
    test_size=0.2,
    random_state=42
)

scaler = StandardScaler()

# O scaler APRENDE média/desvio somente no treino.
X_train_scaled = scaler.fit_transform(X_train)

# O teste usa exatamente a mesma transformação aprendida no treino.
X_test_scaled = scaler.transform(X_test)

print("=== TAMANHOS ===")
print("X_train:", X_train.shape)
print("X_test :", X_test.shape)

print("\n=== TREINO APÓS STANDARDSCALER ===")
print("Média :", X_train_scaled.mean(axis=0).round(3))
print("Desvio:", X_train_scaled.std(axis=0).round(3))

print("\nColunas:", cols)
