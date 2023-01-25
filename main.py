import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def preprocess(raw_data):
    # gestire i NaNs (not a number)
    # trasformare variabili categoriche in numeriche
    # normalizzare i dati
    colonne_selezionate = []
    nomi_colonne_selezionate = []
    for column_name, column in raw_data.items():
        nomi_colonne_selezionate.append(column_name)
        if not column.isna().any():
            colonne_selezionate.append(column.values)
    colonne_numeriche = []
    for colonna in colonne_selezionate:
        if colonna.dtype != int and colonna.dtype != float:
            possibili_valori = list(set(colonna))
            colonne_numeriche.append(
                [possibili_valori.index(elemento) for elemento in colonna]
            )
        else:
            colonne_numeriche.append(colonna)
    colonne_numeriche = np.array(colonne_numeriche)
    colonne_numeriche = colonne_numeriche[1:]
    risultato = []
    for colonna in colonne_numeriche:
        risultato.append((colonna - colonna.min()) / (colonna.max() - colonna.min()))
    risultato = np.array(risultato)
    return risultato 


def train():
    pass


def test():
    pass


def main():
    data_path = os.path.join("data", "train.csv")
    raw_data = pd.read_csv(data_path)
    train_data = preprocess(raw_data)
    print(train_data.shape)

if __name__ == "__main__":
    main()
