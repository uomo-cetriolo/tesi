"""
02_clustering.py
Analisi e segmentazione automatica dei clienti con K-Means (apprendimento non supervisionato).

Passi: 1) caricamento ed esplorazione  2) pulizia  3) normalizzazione
       4) scelta di k (gomito + silhouette)  5) K-Means finale  6) profilo dei gruppi

Input : data/clienti.csv                (creato da 01_genera_dati.py)
Output: data/clienti_con_cluster.csv    clienti + gruppo di appartenenza
        data/profili_cluster.csv        profilo medio di ogni gruppo
        models/kmeans.joblib, models/scaler.joblib   (usati dalla dashboard)
        report/gomito_silhouette.png, report/scatter_cluster.png

Uso:  python src/02_clustering.py
"""

from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")                    # grafici salvati su file, senza finestra
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# --- Parametri -------------------------------------------------------------
SEED = 42
FEATURES = ["n_richieste", "importo_totale", "giorni_ultima_richiesta", "anzianita_giorni"]
K_FINALE = 4                             # scelto guardando gomito e silhouette (vedi grafico)
K_CONFRONTO = [3, 4, 5]                  # configurazioni confrontate
K_RANGE = range(2, 9)

BASE = Path(__file__).resolve().parent.parent
DATA, MODELS, REPORT = BASE / "data", BASE / "models", BASE / "report"
for cartella in (MODELS, REPORT):
    cartella.mkdir(exist_ok=True)


def main():
    # 1) Caricamento ed esplorazione ------------------------------------------
    df = pd.read_csv(DATA / "clienti.csv")
    print(f"Clienti caricati: {len(df)}")
    print("\nValori mancanti:\n", df.isna().sum()[df.isna().sum() > 0].to_string())
    print("\nStatistiche descrittive:\n", df[FEATURES].describe().round(1).to_string())
    print("\nCorrelazioni:\n", df[FEATURES].corr().round(2).to_string())

    # 2) Pulizia di base -------------------------------------------------------
    df = df.drop_duplicates(subset="id_cliente")
    df["zona"] = df["zona"].fillna("Non indicata")         # variabile usata solo nei filtri
    # importo mancante = n. richieste x importo medio per richiesta (mediana dei clienti noti)
    importo_medio = (df["importo_totale"] / df["n_richieste"]).median()
    df["importo_totale"] = df["importo_totale"].fillna((df["n_richieste"] * importo_medio).round(2))
    df["importo_medio_richiesta"] = (df["importo_totale"] / df["n_richieste"]).round(2)

    # 3) Normalizzazione: media 0 e deviazione standard 1 per ogni variabile -----
    scaler = StandardScaler()
    X = scaler.fit_transform(df[FEATURES])

    # 4) Scelta di k: metodo del gomito (inerzia) e indice di silhouette ---------
    risultati = []
    for k in K_RANGE:
        km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(X)
        risultati.append((k, km.inertia_, silhouette_score(X, km.labels_)))
    ris = pd.DataFrame(risultati, columns=["k", "inerzia", "silhouette"])
    print("\nGomito e silhouette per k:\n", ris.round(3).to_string(index=False))
    print("\nConfronto configurazioni:\n", ris[ris["k"].isin(K_CONFRONTO)].round(3).to_string(index=False))

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    a1.plot(ris["k"], ris["inerzia"], marker="o")
    a1.set(title="Metodo del gomito", xlabel="Numero di gruppi (k)", ylabel="Inerzia")
    a2.plot(ris["k"], ris["silhouette"], marker="o", color="tab:orange")
    a2.set(title="Indice di silhouette", xlabel="Numero di gruppi (k)", ylabel="Silhouette")
    plt.tight_layout()
    plt.savefig(REPORT / "gomito_silhouette.png", dpi=150)
    plt.close()

    # 5) K-Means finale ---------------------------------------------------------
    kmeans = KMeans(n_clusters=K_FINALE, n_init=10, random_state=SEED).fit(X)
    df["cluster"] = kmeans.labels_
    silhouette = silhouette_score(X, kmeans.labels_)
    print(f"\nK-Means finale: k = {K_FINALE}, silhouette = {silhouette:.3f}")

    # 6) Profilo dei gruppi -----------------------------------------------------
    profilo = df.groupby("cluster").agg(
        numerosita=("id_cliente", "count"),
        n_richieste=("n_richieste", "mean"),
        importo_totale=("importo_totale", "mean"),
        importo_medio_richiesta=("importo_medio_richiesta", "mean"),
        giorni_ultima_richiesta=("giorni_ultima_richiesta", "mean"),
        anzianita_giorni=("anzianita_giorni", "mean"),
    ).round(1)
    profilo["percentuale"] = (100 * profilo["numerosita"] / len(df)).round(1)
    profilo["nome"] = nomina_gruppi(profilo)

    df["nome_cluster"] = df["cluster"].map(profilo["nome"])
    print("\nProfilo dei gruppi:\n", profilo.to_string())

    # Grafici a dispersione dei gruppi (due coppie di variabili)
    fig, assi = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, x in zip(assi, ["n_richieste", "anzianita_giorni"]):
        for c, g in df.groupby("cluster"):
            ax.scatter(g[x], g["giorni_ultima_richiesta"], s=15, alpha=0.7, label=profilo.loc[c, "nome"])
        ax.set(xlabel=x.replace("_", " "), ylabel="giorni ultima richiesta")
    assi[0].legend()
    fig.suptitle("Gruppi di clienti")
    plt.tight_layout()
    plt.savefig(REPORT / "scatter_cluster.png", dpi=150)
    plt.close()

    # Salvataggio risultati e modello
    df.to_csv(DATA / "clienti_con_cluster.csv", index=False)
    profilo.reset_index().to_csv(DATA / "profili_cluster.csv", index=False)
    joblib.dump(kmeans, MODELS / "kmeans.joblib")
    joblib.dump(scaler, MODELS / "scaler.joblib")
    print("\nFile salvati in data/, models/ e report/")


def nomina_gruppi(profilo):
    """Assegna un nome di marketing a ogni gruppo in base alle sue medie (valido per k = 4)."""
    if len(profilo) != 4:
        return [f"Gruppo {c}" for c in profilo.index]
    nomi, resto = {}, profilo.copy()
    for nome, colonna, scelta in [
        ("Molto attivi", "n_richieste", "idxmax"),                        # tante richieste
        ("Inattivi (a rischio abbandono)", "giorni_ultima_richiesta", "idxmax"),  # fermi da più tempo
        ("Recenti", "anzianita_giorni", "idxmin"),                        # iscritti da poco
    ]:
        gruppo = getattr(resto[colonna], scelta)()
        nomi[gruppo] = nome
        resto = resto.drop(index=gruppo)
    nomi[resto.index[0]] = "Occasionali"                                  # il gruppo che resta
    return pd.Series(nomi)


if __name__ == "__main__":
    main()
