import pandas as pd
import numpy as np
import ipdb


"""

- suddividere in serie di macchine oppure sulla base dei motori che le macchine hanno
- fare una stima del prezzo di queste commponenti nel futuro sulla base di una ricerca su internet

"""

def preprocess(dataset: pd.DataFrame):
    y = dataset.selling_price.values
    # gestire i nans
    # categoriche -> numeriche
    numerical = []
    for (column_name, column) in dataset.items():
        if not column_name == 'selling_price':
            if column_name == 'mileage':
                num_values = []
                for value in column.values:
                    if not type(value) == str and np.isnan(value): # TODO: doesnt work perche' non si puo' controllare se str e' nan
                        num_value = value
                    elif 'kmpl' in value:
                        num_value = float(value.replace(' kmpl', ""))
                    elif 'km/kg' in value:
                        num_value = float(value.replace(' km/kg', ""))*1.4
                    assert type(num_value) == float, ipdb.set_trace()
                    num_values.append(num_value)
                numerical.append(num_values)
            elif column.dtype not in (float, int):
                values = column.values.astype(str)
                try:
                    unique = np.unique(values).tolist()
                except:
                    ipdb.set_trace()
                num_values = []
                for element in values:
                    num_values.append(unique.index(element))
                numerical.append(num_values)
            else:
                numerical.append(column.values)
    numerical = np.array(numerical)
        
    """
    except expression as identifier:
        pass
    cleaned_numerical = []
    for riga in numerical:
        skip = False
        for value in riga:
            if str(value) == 'nan':
                skip = True 
        if not skip:
            cleaned_numerical.append(riga)
    cleaned_numerical = np.array(cleaned_numerical)
    """
    bools = np.isnan(numerical)
    rows_without_nans = bools.sum(axis=0) == 0
    clean_numerical = numerical[:, rows_without_nans]
    # (x - min(x))/(max(x) - min(x))
    col_min = clean_numerical.min(axis=1)
    col_max = clean_numerical.max(axis=1)

    norm_values = (clean_numerical - col_min[:, None])/(
            col_max[:, None] - col_min[:, None]
    )
    assert norm_values.max() == 1.0, "Normalization error."
    assert norm_values.min() == 0.0, "Normalization error."
    return norm_values, y

def train():
    pass


def test():
    pass


def main():
    car_details = pd.read_csv("Car details v3.csv")
    # print(f"{car_details.columns=} {len(car_details)=}")
    preprocessed, y = preprocess(car_details)
    # suddividere i nostri dati in train e test set
    # fittare qualche modello
    # e testare qualche modello
    # guardare le metriche


if __name__ == "__main__":
    main()
