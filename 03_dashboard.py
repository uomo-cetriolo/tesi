"""
03_dashboard.py
Dashboard interattiva (Streamlit) per esplorare i gruppi di clienti trovati dal K-Means
e per classificare un nuovo cliente.

Prima di avviarla servono i file creati dagli script 01 e 02.

Uso (dalla cartella principale del progetto):
    streamlit run src/03_dashboard.py
"""

from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.metrics import silhouette_score

BASE = Path(__file__).resolve().parent.parent
DATA, MODELS = BASE / "data", BASE / "models"

# Variabili numeriche: colonna -> etichetta mostrata
VARIABILI = {
    "n_richieste": "Numero di richieste",
    "importo_totale": "Importo totale speso (€)",
    "importo_medio_richiesta": "Importo medio per richiesta (€)",
    "giorni_ultima_richiesta": "Giorni dall'ultima richiesta",
    "anzianita_giorni": "Anzianità (giorni)",
}
# Variabili categoriche (non usate dal K-Means): colonna -> etichetta
CATEGORICHE = {
    "categoria_preferita": "Categoria di servizio",
    "zona": "Zona",
    "periodo_settimana": "Periodo della settimana",
    "fascia_oraria": "Fascia oraria",
}
# Azione di marketing suggerita per ogni gruppo
AZIONI = {
    "Molto attivi": "Programma fedeltà, accesso anticipato ai nuovi servizi, offerte premium.",
    "Occasionali": "Promozioni mirate e promemoria per aumentare la frequenza di richiesta.",
    "Recenti": "Percorso di benvenuto, sconto sul secondo ordine, guida ai servizi.",
    "Inattivi (a rischio abbandono)": "Campagna di riattivazione con offerta a tempo limitato.",
}


# --- Caricamento dati e modello --------------------------------------------
@st.cache_data
def carica_dati():
    df = pd.read_csv(DATA / "clienti_con_cluster.csv", parse_dates=["data_iscrizione"])
    profili = pd.read_csv(DATA / "profili_cluster.csv")
    return df, profili


@st.cache_resource
def carica_modello():
    return joblib.load(MODELS / "kmeans.joblib"), joblib.load(MODELS / "scaler.joblib")


@st.cache_data
def calcola_silhouette(df, _scaler):
    """Indice di silhouette dei gruppi su tutti i clienti (tra -1 e 1)."""
    X = _scaler.transform(df[list(_scaler.feature_names_in_)])
    return silhouette_score(X, df["cluster"])


# --- Funzioni di calcolo ---------------------------------------------------
def applica_filtri(df, zone, categorie, inizio, fine):
    """Tiene i clienti con zona e categoria scelte, iscritti nel periodo indicato."""
    maschera = (
        df["zona"].isin(zone)
        & df["categoria_preferita"].isin(categorie)
        & df["data_iscrizione"].between(pd.Timestamp(inizio), pd.Timestamp(fine))
    )
    return df[maschera]


def classifica_cliente(valori, kmeans, scaler):
    """Normalizza i valori di un nuovo cliente e restituisce il numero del suo gruppo."""
    X = pd.DataFrame([valori])[list(scaler.feature_names_in_)]
    return int(kmeans.predict(scaler.transform(X))[0])


# --- Interfaccia -----------------------------------------------------------
def main():
    st.set_page_config(page_title="Segmentazione clienti", layout="wide")
    st.title("Segmentazione automatica della clientela")

    if not (DATA / "clienti_con_cluster.csv").exists() or not (MODELS / "kmeans.joblib").exists():
        st.error("File non trovati. Esegui prima: python src/01_genera_dati.py e python src/02_clustering.py")
        st.stop()

    df, profili = carica_dati()
    kmeans, scaler = carica_modello()
    nome_gruppo = profili.set_index("cluster")["nome"]
    colori = dict(zip(profili["nome"], px.colors.qualitative.Set2))

    # Filtri nella barra laterale
    st.sidebar.header("Filtri")
    zone = st.sidebar.multiselect("Zona", sorted(df["zona"].unique()), default=sorted(df["zona"].unique()))
    categorie = st.sidebar.multiselect(
        "Categoria di servizio preferita",
        sorted(df["categoria_preferita"].unique()),
        default=sorted(df["categoria_preferita"].unique()),
    )
    d_min, d_max = df["data_iscrizione"].min().date(), df["data_iscrizione"].max().date()
    periodo = st.sidebar.date_input(
        "Periodo di iscrizione", value=(d_min, d_max), min_value=d_min, max_value=d_max
    )
    if len(periodo) != 2:
        st.info("Seleziona la data di inizio e di fine del periodo.")
        st.stop()

    df_f = applica_filtri(df, zone, categorie, periodo[0], periodo[1])
    if df_f.empty:
        st.warning("Nessun cliente corrisponde ai filtri scelti.")
        st.stop()

    # Indicatori principali
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Clienti selezionati", f"{len(df_f)} su {len(df)}")
    c2.metric("Gruppi individuati", df["cluster"].nunique())
    c3.metric("Indice di silhouette", f"{calcola_silhouette(df, scaler):.2f}")
    c4.metric("Spesa media per cliente", f"{df_f['importo_totale'].mean():.0f} €")

    tab_gruppi, tab_profili, tab_nuovo, tab_dati = st.tabs(
        ["Gruppi", "Profili", "Nuovo cliente", "Dati"]
    )

    # --- Tab 1: scatter plot e dimensione dei gruppi
    with tab_gruppi:
        sx, sy = st.columns(2)
        asse_x = sx.selectbox("Asse X", list(VARIABILI), index=0, format_func=VARIABILI.get)
        asse_y = sy.selectbox("Asse Y", list(VARIABILI), index=3, format_func=VARIABILI.get)
        fig = px.scatter(
            df_f, x=asse_x, y=asse_y, color="nome_cluster", color_discrete_map=colori,
            hover_data=["id_cliente", "zona", "categoria_preferita"],
            labels={**VARIABILI, "nome_cluster": "Gruppo"}, opacity=0.75,
        )
        st.plotly_chart(fig)

        sinistra, destra = st.columns(2)
        conteggio = df_f["nome_cluster"].value_counts().reset_index()
        conteggio.columns = ["Gruppo", "Clienti"]
        sinistra.subheader("Numerosità dei gruppi")
        sinistra.plotly_chart(px.bar(
            conteggio, x="Gruppo", y="Clienti", color="Gruppo", color_discrete_map=colori,
        ))
        destra.subheader("Composizione dei gruppi")
        variabile = destra.selectbox("Variabile", list(CATEGORICHE), format_func=CATEGORICHE.get)
        destra.plotly_chart(px.histogram(
            df_f, x=variabile, color="nome_cluster", color_discrete_map=colori, barmode="group",
            labels={**CATEGORICHE, "nome_cluster": "Gruppo"},
        ))

    # --- Tab 2: profilo di ciascun gruppo (calcolato sui clienti filtrati)
    with tab_profili:
        st.subheader("Profilo medio di ciascun gruppo")
        tabella = df_f.groupby("nome_cluster")[list(VARIABILI)].mean().round(1)
        tabella.insert(0, "n_clienti", df_f.groupby("nome_cluster").size())
        tabella = tabella.rename(columns={**VARIABILI, "n_clienti": "Clienti"})
        tabella["Azione di marketing consigliata"] = [AZIONI.get(n, "-") for n in tabella.index]
        st.dataframe(tabella.reset_index().rename(columns={"nome_cluster": "Gruppo"}), hide_index=True)

    # --- Tab 3: classificazione di un nuovo cliente
    with tab_nuovo:
        st.subheader("A quale gruppo appartiene un nuovo cliente?")
        with st.form("nuovo_cliente"):
            a, b = st.columns(2)
            n_richieste = a.number_input("Numero di richieste", min_value=1, value=5, step=1)
            importo = b.number_input("Importo totale speso (€)", min_value=1.0, value=400.0, step=10.0)
            giorni = a.number_input("Giorni dall'ultima richiesta", min_value=0, value=20, step=1)
            anzianita = b.number_input("Anzianità (giorni dall'iscrizione)", min_value=0, value=120, step=1)
            invia = st.form_submit_button("Individua il gruppo")

        if invia:
            if giorni > anzianita:
                st.error("I giorni dall'ultima richiesta non possono superare l'anzianità del cliente.")
            else:
                valori = {
                    "n_richieste": n_richieste, "importo_totale": importo,
                    "giorni_ultima_richiesta": giorni, "anzianita_giorni": anzianita,
                }
                nome = nome_gruppo[classifica_cliente(valori, kmeans, scaler)]
                st.success(f"Il cliente appartiene al gruppo: {nome}")
                st.write(AZIONI.get(nome, ""))
                medie = profili.set_index("nome").loc[nome]
                confronto = pd.DataFrame({
                    "Nuovo cliente": [n_richieste, importo, giorni, anzianita],
                    "Media del gruppo": [medie["n_richieste"], medie["importo_totale"],
                                         medie["giorni_ultima_richiesta"], medie["anzianita_giorni"]],
                }, index=[VARIABILI[c] for c in
                          ["n_richieste", "importo_totale", "giorni_ultima_richiesta", "anzianita_giorni"]])
                st.dataframe(confronto)

    # --- Tab 4: dati e download
    with tab_dati:
        st.dataframe(df_f.drop(columns="cluster"), hide_index=True)
        st.download_button("Scarica i dati filtrati (CSV)", df_f.to_csv(index=False), "clienti_filtrati.csv")


if __name__ == "__main__":
    main()
