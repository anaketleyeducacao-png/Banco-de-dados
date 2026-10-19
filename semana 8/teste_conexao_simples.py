import os
from sqlalchemy import create_engine, text

url = os.getenv("DATABASE_URL")

if not url:
    raise RuntimeError("DATABASE_URL não foi carregada neste terminal.")

engine = create_engine(url)

with engine.connect() as conn:
    linha = conn.execute(
        text("SELECT current_database(), current_user")
    ).one()

print("Banco conectado:", linha[0])
print("Usuário:", linha[1])
print("CONEXÃO OK")