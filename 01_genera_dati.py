"""
01_genera_dati.py
Genera 500 clienti INVENTATI di una startup di servizi su richiesta
e li salva in data/clienti.csv (una riga per cliente, nessun dato personale).

Uso:  python src/01_genera_dati.py
"""

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# --- Parametri -------------------------------------------------------------
SEED = 42                                # stesso seed = stessi numeri
N_CLIENTI = 500
GIORNI_PERIODO = 180                     # "ultimo periodo" = ultimi 180 giorni
DATA_INIZIO = date(2026, 1, 1)           # data di partenza della startup (prima iscrizione possibile)
DATA_RIFERIMENTO = date.today()          # "oggi"; per una data fissa: date(2026, 11, 21)
PERC_MANCANTI = 0.02                     # quota di valori mancanti (0 = nessuno)
OUTPUT_CSV = Path(__file__).resolve().parent.parent / "data" / "clienti.csv"

# --- Variabili categoriche -------------------------------------------------
ZONE = ["Nord", "Centro", "Sud", "Isole"]
PERIODI = ["Infrasettimanale", "Weekend"]
FASCE = ["Mattina", "Pomeriggio", "Sera", "Pasti"]
PREZZO_CATEGORIA = {                     # categoria: moltiplicatore del prezzo
    "Pronto intervento": 1.2,
    "Benessere": 1.1,
    "Consulenze": 1.5,
}

# --- Profili dei clienti ---------------------------------------------------
# profilo: (anzianità min-max (giorni dall'iscrizione),
#           giorni dall'ultima richiesta min-max,
#           numero di richieste effettuate min-max,
#           importo medio speso per richiesta, in euro)
PROFILI = {
    "molto_attivo": ((90, 365),  (0, 7),    (20, 50), 70),
    "occasionale":  ((60, 365),  (10, 50),  (2, 8),   70),
    "recente":      ((0, 14),    (0, 7),    (1, 4),   70),
    "inattivo":     ((150, 365), (70, 170), (1, 4),   70),
}


def main():
    rng = np.random.default_rng(SEED)
    nomi = list(PROFILI)
    categorie = list(PREZZO_CATEGORIA)
    giorni_totali = (DATA_RIFERIMENTO - DATA_INIZIO).days   # anzianità massima possibile

    righe = []
    for i in range(N_CLIENTI):
        # 1) scelgo il profilo del cliente
        (a_min, a_max), (u_min, u_max), (r_min, r_max), importo_tipico = PROFILI[rng.choice(nomi)]

        # 2) data di iscrizione (mai prima della partenza) e anzianità in giorni reali
        a_max = min(a_max, giorni_totali)
        a_min = min(a_min, a_max)
        data_iscrizione = DATA_RIFERIMENTO - timedelta(days=int(rng.integers(a_min, a_max + 1)))
        anzianita = (DATA_RIFERIMENTO - data_iscrizione).days

        # 3) giorni dall'ultima richiesta: non può precedere l'iscrizione
        finestra = min(anzianita, GIORNI_PERIODO)
        giorni_ultima = rng.integers(min(u_min, finestra), min(u_max, finestra) + 1)

        # 4) numero di richieste effettuate
        n_richieste = rng.integers(r_min, r_max + 1)

        # 5) importo totale = n. richieste x importo medio (dipende dalla categoria)
        categoria = rng.choice(categorie)
        importo_medio = rng.lognormal(np.log(importo_tipico * PREZZO_CATEGORIA[categoria]), 0.3)

        righe.append({
            "id_cliente": f"CLI-{i + 1:04d}",
            "zona": rng.choice(ZONE),
            "data_iscrizione": data_iscrizione,
            "n_richieste": n_richieste,
            "importo_totale": round(n_richieste * importo_medio, 2),
            "giorni_ultima_richiesta": giorni_ultima,
            "categoria_preferita": categoria,
            "periodo_settimana": rng.choice(PERIODI),
            "fascia_oraria": rng.choice(FASCE),
        })

    df = pd.DataFrame(righe)

    # Controllo di coerenza: nessuna richiesta prima dell'iscrizione
    anzianita_giorni = (pd.Timestamp(DATA_RIFERIMENTO) - pd.to_datetime(df["data_iscrizione"])).dt.days
    assert (df["giorni_ultima_richiesta"] <= anzianita_giorni).all()

    # Qualche valore mancante, per mostrare poi la sua gestione nell'analisi
    for colonna in ["zona", "importo_totale"]:
        df.loc[rng.random(len(df)) < PERC_MANCANTI, colonna] = np.nan

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"Generati {len(df)} clienti -> {OUTPUT_CSV}\n")
    print(df.head())
    print(df[["n_richieste", "importo_totale", "giorni_ultima_richiesta"]].describe().round(1))


if __name__ == "__main__":
    main()
