from sentence_transformers import SentenceTransformer, util

modelo = SentenceTransformer(
    "paraphrase-multilingual-MiniLM-L12-v2"
)

# A é a frase de referência.
A = "Normalização remove redundância em banco de dados."

# B, C e D serão comparadas com A.
B = "Normalizar tabelas ajuda a evitar dados repetidos."
C = "JOIN combina linhas de tabelas relacionadas."
D = "Bolo de chocolate leva farinha, ovos e açúcar."

frases = [A, B, C, D]

embeddings = modelo.encode(frases, convert_to_tensor=True)

scores = util.cos_sim(
    embeddings[0],      # vetor de A
    embeddings[1:]      # vetores de B, C e D
)[0]

print("A =", A)

for letra, frase, score in zip(
    ["B", "C", "D"],
    [B, C, D],
    scores
):
    print(f"{letra} = {score.item():.4f} | {frase}")