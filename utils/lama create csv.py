import os

import pickle
import pandas as pd
import kagglehub
import pyarrow.parquet as pq
import gc
from pathlib import Path
import sys
from catboost import CatBoostClassifier, Pool
import numpy as np
import lightautoml


sys.path.append(os.path.join(os.path.dirname(__file__), '../..', 'utils'))

from target_constructor import target_constructor # type: ignore
from comp_db_dwld import process_incremental # type: ignore


# === 1. Чтение данных ===
def read(product_id):
    path = kagglehub.dataset_download("alexxl/sbol-dataset")

    df = process_incremental(path + '/sbolpro_merged_final.pqt', product_id)
    return df


# === 2. Топ-200 признаков по важности ===
def load_top_features(prod_id, k=200):
    # Путь к корню проекта (personalized-offers-LGBM)
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "./.."))

    imp_path = os.path.join(PROJECT_ROOT, "utils", 'best_feature', f"rfe_metrics_history_{prod_id}.csv")

    if not os.path.exists(imp_path):
        raise FileNotFoundError(f"Файл не найден: {imp_path}")

    imp = pd.read_csv(imp_path)['dropped_feature'].dropna().astype(str).tolist()

    return imp


def read_model(model_name, product_id):
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "./.."))
    model_filename = f"{model_name}_{product_id}.pkl"
    model_path = os.path.join(PROJECT_ROOT, model_filename)

    with open(model_path, 'rb') as f:
        return pickle.load(f)




print('okay, lets go')
models = ['lama']
products = [0, 1, 2, 4, 5, 6]
one_id = 0
df = read(one_id)
print('finally, we have red data')
df_list = []
for product_id in products:
    print('now: product', product_id)
    _, _, x_oot, _ = target_constructor(df, product_id)
    drop_cols = ['timestamp', 'product_id']
    x_oot = x_oot.drop(drop_cols, axis=1)
    df_list.append(x_oot)
df = pd.concat(df_list, ignore_index=True)
df = df.drop_duplicates(subset=['user_id'], keep='last')

for product_id in products:
    print(f'Start product {product_id}')
    top_feats = load_top_features(product_id)
    df_product = df[['user_id'] + top_feats]
    df_product = df_product.fillna(0)
    X = df_product.drop('user_id', axis=1)
    predictions_dict = {'user_id': df_product['user_id'].values}
    for model_name in models:
        print(f'now {model_name} works. good luck!')
        model = read_model(model_name, product_id)       
            
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X)
            # Берем вероятность положительного класса (обычно второй столбец)
            if proba.ndim == 2:
                predictions = proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]
            else:
                predictions = proba
        elif hasattr(model, 'predict'):
            predictions = model.predict(X)
        else:
            print(f"    Модель {model_name} не имеет метода predict_proba или predict")
            predictions = np.zeros(len(df))
        predictions_dict[f'{model_name}_proba'] = predictions
        print(f'wow. product {product_id} with {model_name} done!')
    df_predictions = pd.DataFrame(predictions_dict)
    columns_order = ['user_id', 'product_id', 'catboost_proba', 'lama_proba']
    columns_order = [col for col in columns_order if col in df_predictions.columns]
    output_path = f"product_{product_id}.csv"
    df_predictions.to_csv(output_path, index=False)
    print(f"\nproduct {product_id} data are saved to {output_path}")
    
    
