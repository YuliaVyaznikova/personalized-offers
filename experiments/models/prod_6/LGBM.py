
import kagglehub
import sys
import os
import pandas as pd
from lightgbm import LGBMClassifier

import optuna
import os
import category_encoders as ce

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..', 'utils'))

from target_constructor import target_constructor # type: ignore
from comp_db_dwld import process_incremental # type: ignore
from type_of_feature import type_of_feature # type: ignore
from metric_report import metric_report # type: ignore
from optunaizer import objective # type: ignore


path = kagglehub.dataset_download("alexxl/sbol-dataset")
print("Path to dataset files:", path)
df = process_incremental(path+'/'+ 'sbolpro_merged_final.pqt')
print(df.columns)
x_train, y_train, x_oot, y_oot = target_constructor(df, 6)#делим 

#соединяем для того чтобы разделить на валидации
df_train_combined = x_train.copy()
df_train_combined['is_sold'] = y_train

x_tr, y_tr, x_val, y_val = target_constructor(df_train_combined, 6)# для валидации

#удаляем ненужные фичи
x_tr = x_tr.drop(['timestamp', 'user_id', 'product_id', 'year_month'], axis=1)
x_val = x_val.drop(['timestamp', 'user_id', 'product_id', 'year_month'], axis=1)

x_train = x_train.drop(['timestamp', 'user_id', 'product_id', 'year_month'], axis=1)
x_oot = x_oot.drop(['timestamp', 'user_id', 'product_id', 'year_month'], axis=1)

# Деление на числовые и категориальные с оптимизируемым threshold
_, cat_cols = type_of_feature(df, 20)

num_cols = [col for col in x_train.columns if col not in cat_cols]
    
# Удаляем ненужные названия столбцов
cat_cols = cat_cols[:-5]

print(cat_cols)
    
# Создаем study для оптимизации
study = optuna.create_study(
    direction='maximize',  # Максимизируем ROC-AUC
    study_name='lgbm_optimization',
    sampler=optuna.samplers.TPESampler(seed=42)
)
    
print("Начинаем оптимизацию с Optuna...")
print("=" * 50)
    
# Запускаем оптимизацию
study.optimize(lambda trial: objective(trial,x_tr, y_tr, x_val, y_val, cat_cols, num_cols), n_trials=30, show_progress_bar=True)
    
print("\n" + "=" * 50)
print("ОПТИМИЗАЦИЯ ЗАВЕРШЕНА")
print("=" * 50)
    
# Выводим лучшие результаты
print(f"Лучший ROC-AUC: {study.best_value:.4f}")
print("Лучшие параметры:")
for key, value in study.best_params.items():
    print(f"  {key}: {value}")
   
# =============================================
# ОБУЧАЕМ ФИНАЛЬНУЮ МОДЕЛЬ С ЛУЧШИМИ ПАРАМЕТРАМИ
# =============================================
    
print("\n" + "=" * 50)
print("ОБУЧЕНИЕ ФИНАЛЬНОЙ МОДЕЛИ НА ВСЕХ TRAIN ДАННЫХ")
print("=" * 50)
        
        
best_params = study.best_params

# Преобразуем категориальные признаки
target_encoder = ce.TargetEncoder(cols=cat_cols, smoothing=best_params['smoothing'])
x_train_enc = target_encoder.fit_transform(x_train, y_train)
x_oot_enc = target_encoder.transform(x_oot)
    
# Оставляем числовые
x_train_enc[num_cols] = x_train[num_cols]
x_oot_enc[num_cols] = x_oot[num_cols]
    
# Финальная модель с лучшими параметрами
final_model = LGBMClassifier(
    n_estimators=best_params['n_estimators'],
    learning_rate=best_params['learning_rate'],
    num_leaves=best_params['num_leaves'],
    max_depth=best_params['max_depth'],
    min_child_samples=best_params['min_child_samples'],
    subsample=best_params['subsample'],
    colsample_bytree=best_params['colsample_bytree'],
    reg_alpha=best_params['reg_alpha'],
    reg_lambda=best_params['reg_lambda'],
    random_state=42,
    verbose=-1
)
#учим
final_model.fit(x_train_enc, y_train)

#смотрим на результат
y_pred = final_model.predict_proba(x_oot_enc)[:, 1]
metric_report(y_oot, y_pred)


"""
#делим на числове и категориальные
num_cols, cat_cols = type_of_feature(df, 1000)

#удаляем ненужные названия столбцов
cat_cols= cat_cols[3:-1]

#преобразуем категориальные призники с помощьюTargetEncoder
target_encoder = ce.TargetEncoder(cols=cat_cols, smoothing=0.3)
x_train_enc = target_encoder.fit_transform(x_train, y_train)
x_oot_enc = target_encoder.transform(x_oot)

#оставляем числовые
x_train_enc[num_cols] = x_train[num_cols]
x_oot_enc[num_cols] = x_oot[num_cols]

#конфигурация сети
model = LGBMClassifier(
    n_estimators=100,
    learning_rate=0.1,
    num_leaves=31,
    random_state=42
)

#учим
model.fit(x_train_enc, y_train)

#смотрим на результат
y_pred = model.predict_proba(x_oot_enc)[:, 1]
metric_report(y_oot, y_pred)

# тут был 85 72
"""
