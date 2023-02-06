import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


def preprocess(raw_data, raw_test):
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
        min_col = colonna.min()
        max_col = colonna.max()
        risultato.append((colonna - min_col) / (max_col - min_col))
    risultato = np.array(risultato)
    return risultato


def train(model, X, y):
    model.fit(X, y)
    # mean square error (MSE) mean(||y_pred - y||^2)
    # mean absolute error (MAE) mean(|y_pred - y|)
    y_pred = model.predict(X)
    mse = np.mean(((y - y_pred) ** 2))
    mae = np.mean(np.abs(y_pred - y))
    print(f"{mse=:.4f} {mae=:.4f}")
    plt.title(f"{model.__class__.__name__} MSE={mse:.4f} MAE={mae:.4f}")
    plt.scatter(np.arange(len(y)), y, label="real")
    plt.scatter(np.arange(len(y)), y_pred, label="predicted")
    plt.legend()
    plt.show()
    return mse, mae


def test(model, X, y):
    pass


def main():
    data_path = os.path.join("data", "train.csv")
    test_data_path = os.path.join("data", "test.csv")
    raw_data = pd.read_csv(data_path)
    raw_test = pd.read_csv(test_data_path)
    train_data = preprocess(raw_data, raw_test)
    X, y = train_data[:-1].T, train_data[-1]
    model = LinearRegression()
    train(model, X, y)
    # test(model)


if __name__ == "__main__":
    main()
