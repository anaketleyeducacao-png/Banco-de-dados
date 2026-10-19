
# 05_pipeline_completo.py
# Extração -> auditoria -> limpeza -> features -> encoding
# -> split -> StandardScaler -> artefatos.

from dotenv import load_dotenv
from pathlib import Path
load_dotenv(Path(__file__).parent / ".env")
import os
import joblib
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

engine = create_engine(os.environ["DATABASE_URL"])

# 1. EXTRAÇÃO
df = pd.read_sql("""
    SELECT
        c.id AS cliente_id,
        c.cidade,
        c.email AS email,
        COUNT(p.id) AS total_pedidos,
        COALESCE(ROUND(AVG(p.total)::numeric, 2), 0) AS ticket_medio,
        COALESCE(SUM(p.total), 0) AS total_gasto,
        COALESCE(MAX(CURRENT_DATE - p.criado_em::date), 9999) AS dias_inativo,
        COUNT(*) FILTER (WHERE p.status = 'cancelado') AS qtd_cancelados,
        COUNT(*) FILTER (WHERE p.status = 'entregue') AS qtd_entregues
    FROM clientes c
    LEFT JOIN pedidos p ON p.cliente_id = c.id
    GROUP BY c.id, c.cidade, c.email
""", engine)

# 2. AUDITORIA ANTES DA LIMPEZA
print("=== AUDITORIA INICIAL ===")
print("Shape:", df.shape)
print("Nulos:")
print(df.isnull().sum())
print("Duplicatas por cliente_id:", df.duplicated("cliente_id").sum())

# 3. LIMPEZA
df["cidade"] = df["cidade"].fillna("Desconhecida")
df = df.drop_duplicates(subset="cliente_id")

# 4. FEATURE ENGINEERING
df["taxa_cancelamento"] = (
    df["qtd_cancelados"] /
    df["total_pedidos"].replace(0, np.nan)
).fillna(0)

df["cliente_ativo"] = (
    df["dias_inativo"] < 30
).astype(int)

df["faixa_gasto_enc"] = pd.cut(
    df["total_gasto"],
    bins=[-0.01, 500, 2000, 5000, np.inf],
    labels=[0, 1, 2, 3]
).astype(int)

# 5. LABEL ENCODING DA CIDADE
le = LabelEncoder()
df["cidade_enc"] = le.fit_transform(df["cidade"])
joblib.dump(le, "le_cidade_v1.pkl")

# 6. TARGET HIPOTÉTICO
df["churn"] = (
    df["dias_inativo"] > 60
).astype(int)

feature_cols = [
    "total_pedidos",
    "ticket_medio",
    "total_gasto",
    "dias_inativo",
    "taxa_cancelamento",
    "cliente_ativo",
    "faixa_gasto_enc",
    "cidade_enc"
]

X = df[feature_cols].astype(float)
y = df["churn"]

# 7. SPLIT ANTES DO SCALER
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# 8. STANDARD SCALER
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)
joblib.dump(scaler, "scaler_v1.pkl")

# 9. EXPORTAÇÃO
pd.DataFrame(
    X_train_s,
    columns=feature_cols
).to_parquet("X_train.parquet", index=False)

pd.DataFrame(
    X_test_s,
    columns=feature_cols
).to_parquet("X_test.parquet", index=False)

y_train.to_frame().to_parquet(
    "y_train.parquet",
    index=False
)

y_test.to_frame().to_parquet(
    "y_test.parquet",
    index=False
)

print("\n=== RESULTADO FINAL ===")
print("Treino:", X_train_s.shape)
print("Teste :", X_test_s.shape)
print("Churn treino:", f"{y_train.mean():.1%}")

print("\nArtefatos gerados:")
for nome in [
    "X_train.parquet",
    "X_test.parquet",
    "y_train.parquet",
    "y_test.parquet",
    "scaler_v1.pkl",
    "le_cidade_v1.pkl"
]:
    print("-", nome)
