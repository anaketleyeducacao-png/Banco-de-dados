
# 02_encoding_status.py
# Este exercício usa exatamente a coluna STATUS da tabela pedidos.
from dotenv import load_dotenv
from pathlib import Path
load_dotenv(Path(__file__).parent / ".env")
import os
import pandas as pd
from sqlalchemy import create_engine
from sklearn.preprocessing import LabelEncoder

engine = create_engine(os.environ["DATABASE_URL"])

df_status = pd.read_sql("""
    SELECT status
    FROM pedidos
    WHERE status IS NOT NULL
""", engine)

print("=== STATUS ORIGINAL ===")
print(df_status["status"].value_counts())

# 1. LABEL ENCODING
le = LabelEncoder()
df_status["status_label"] = le.fit_transform(df_status["status"])

mapeamento = dict(
    zip(le.classes_, le.transform(le.classes_))
)

print("\n=== LABEL ENCODING ===")
print(mapeamento)
print(df_status.head(10))

# 2. ONE-HOT ENCODING
df_ohe = pd.get_dummies(
    df_status[["status"]],
    columns=["status"],
    prefix="status",
    dtype=int
)

print("\n=== ONE-HOT ENCODING ===")
print("Colunas criadas:", list(df_ohe.columns))
print(df_ohe.head(10))

print("\n=== COMPARAÇÃO ===")
print("Label: uma única coluna numérica.")
print("One-Hot: uma coluna 0/1 para cada categoria.")
