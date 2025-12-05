import uuid
import kagglehub
import sys
import os
from datetime import datetime
from catboost import CatBoostClassifier, Pool
import optuna
import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '../..', 'utils'))

from target_constructor import target_constructor # type: ignore
from comp_db_dwld import process_incremental # type: ignore
from type_of_feature import type_of_feature # type: ignore
from metric_report import metric_report # type: ignore
from optunaizer_CatBoost import objective_catboost # type: ignore
from out_file import write_to_file # type: ignore
from out_file import parse_args # type: ignore

class CatBoostPipeline:
    def __init__(self, product_id, n_trials, threshold, n_features_to_keep=200):
        self.product_id = product_id
        self.n_trials = n_trials
        self.threshold = threshold
        self.filepath = f'../../artifacts/prod_{product_id}/CatBoost_{datetime.now().strftime("%Y%m%d_%H%M%S")}_{str(uuid.uuid4())[:8]}.txt'
        
        self.model = None
        self.best_params = None
        self.cat_features_indices = None
        self.n_features_to_keep = n_features_to_keep

        
    def read(self):        
        path = kagglehub.dataset_download("alexxl/sbol-dataset")
        df = process_incremental(path + '/sbolpro_merged_final.pqt', self.product_id)
        
        write_to_file(self.filepath, f"Размер исходных данных: {df.shape}")
        write_to_file(self.filepath, f"Колонки: {list(df.columns)}")
        
        return df
    
    
    def select_features(self, x_tr, y_tr, cat_features, n_features=200):
        """Отбор наиболее важных фичей с помощью быстрой CatBoost модели"""
        
        write_to_file(self.filepath, f"\n=== Отбор фичей ===")
        write_to_file(self.filepath, f"Исходное количество фичей: {x_tr.shape[1]}")
        write_to_file(self.filepath, f"Целевое количество фичей: {n_features}")
        
        # Убедимся, что категориальные признаки преобразованы
        x_tr_prepared = x_tr.copy()
        self._prepare_categorical_features(x_tr_prepared, cat_features)
        
        # Создаем временную модель для оценки важности фичей
        cat_indices = [i for i, col in enumerate(x_tr_prepared.columns) if col in cat_features]
        
        # Проверяем, что все категориальные признаки действительно строки
        for idx in cat_indices:
            col_name = x_tr_prepared.columns[idx]
            unique_vals = x_tr_prepared[col_name].unique()[:5]
            write_to_file(self.filepath, f"  Категориальный признак '{col_name}': примеры значений {list(unique_vals)}")
        
        train_pool = Pool(
            data=x_tr_prepared,
            label=y_tr,
            cat_features=cat_indices
        )
        
        # Быстрая модель для оценки важности
        temp_model = CatBoostClassifier(
            iterations=200,
            depth=6,
            learning_rate=0.1,
            verbose=False,
            random_seed=42,
            allow_writing_files=False,
            task_type='CPU'
        )
        
        try:
            temp_model.fit(train_pool)
        except Exception as e:
            write_to_file(self.filepath, f"Ошибка при обучении временной модели: {e}")
            write_to_file(self.filepath, "Пробуем без категориальных признаков...")
            
            # Если ошибка, пробуем без категориальных признаков
            train_pool = Pool(data=x_tr_prepared, label=y_tr)
            temp_model.fit(train_pool)
            
            # Важность будет только для некатегориальных признаков
            cat_features = []
            cat_indices = []
        
        # Получаем важность фичей
        importances = temp_model.get_feature_importance()
        importance_df = pd.DataFrame({
            'feature': x_tr_prepared.columns,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        # Выбираем топ-N фичей
        selected_columns = importance_df.head(n_features)['feature'].tolist()
        self.selected_features = selected_columns
        
        # Сохраняем информацию о важности
        write_to_file(self.filepath, f"Выбрано фичей: {len(selected_columns)}")
        write_to_file(self.filepath, "Топ-20 самых важных фичей:")
        for i, row in importance_df.head(20).iterrows():
            write_to_file(self.filepath, f"  {row['feature']}: {row['importance']:.4f}")
        
        # Сохраняем полный список важности в файл
        importance_file = f'../../artifacts/prod_{self.product_id}/feature_importance_initial.csv'
        importance_df.to_csv(importance_file, index=False)
        write_to_file(self.filepath, f"Важность фичей сохранена в: {importance_file}")
        
        return selected_columns
    
    
    def select(self, df):
        """Разделение данных и выбор фичей"""
        
        # Разделение на train и OOT
        x_train, y_train, x_oot, y_oot = target_constructor(df, self.product_id)
        
        # Создание валидационного набора
        df_train_combined = x_train.copy()
        df_train_combined['is_sold'] = y_train
        x_tr, y_tr, x_val, y_val = target_constructor(df_train_combined, self.product_id)
        
        # Удаление служебных колонок
        drop_cols = ['timestamp', 'user_id', 'product_id', 'year_month']
        
        x_tr = x_tr.drop(drop_cols, axis=1)
        x_val = x_val.drop(drop_cols, axis=1)
        x_train = x_train.drop(drop_cols, axis=1)
        x_oot = x_oot.drop(drop_cols, axis=1)
        
        # Определение типов фичей для CatBoost
        _, cat_cols = type_of_feature(df, self.threshold)
        cat_cols = cat_cols[:-5]  # Удаление последних 5 колонок
        cat_cols = [col for col in cat_cols if col in x_train.columns]
        
        # ПРЕОБРАЗОВАНИЕ КАТЕГОРИАЛЬНЫХ ПРИЗНАКОВ
        # CatBoost требует, чтобы категориальные признаки были строковыми или целочисленными
        self._prepare_categorical_features(x_tr, cat_cols)
        self._prepare_categorical_features(x_val, cat_cols)
        self._prepare_categorical_features(x_train, cat_cols)
        self._prepare_categorical_features(x_oot, cat_cols)
        
        write_to_file(self.filepath, f"\nПосле преобразования категориальных признаков:")
        write_to_file(self.filepath, f"Размер x_tr: {x_tr.shape}")
        write_to_file(self.filepath, f"Размер x_val: {x_val.shape}")
        
        # ОТБОР ФИЧЕЙ
        if self.n_features_to_keep and x_tr.shape[1] > self.n_features_to_keep:
            selected_features = self.select_features(
                x_tr, y_tr, cat_cols, 
                n_features=self.n_features_to_keep
            )
            
            # Оставляем только выбранные фичи во всех наборах
            x_tr = x_tr[selected_features]
            x_val = x_val[selected_features]
            x_train = x_train[selected_features]
            x_oot = x_oot[selected_features]
            
            # Обновляем список категориальных признаков
            cat_cols = [col for col in cat_cols if col in selected_features]
        
        # Определение индексов категориальных признаков
        all_columns = list(x_train.columns)
        self.cat_features_indices = [all_columns.index(col) for col in cat_cols if col in all_columns]
        
        write_to_file(self.filepath, f"\nИтоговые размеры данных:")
        write_to_file(self.filepath, f"Размер x_tr: {x_tr.shape}")
        write_to_file(self.filepath, f"Размер x_val: {x_val.shape}")
        write_to_file(self.filepath, f"Размер x_train: {x_train.shape}")
        write_to_file(self.filepath, f"Размер x_oot: {x_oot.shape}")
        write_to_file(self.filepath, f"Категориальные фичи ({len(cat_cols)}): {cat_cols[:10]}...")
        write_to_file(self.filepath, f"Индексы категориальных фичей: {self.cat_features_indices[:10]}...")
        
        return {
            'x_tr': x_tr, 'y_tr': y_tr,
            'x_val': x_val, 'y_val': y_val,
            'x_train': x_train, 'y_train': y_train,
            'x_oot': x_oot, 'y_oot': y_oot,
            'cat_features': cat_cols
        }
    
    def _prepare_categorical_features(self, df, cat_cols):
        """Преобразование категориальных признаков для CatBoost"""
        for col in cat_cols:
            if col in df.columns:
                # Преобразуем все значения в строки
                df[col] = df[col].astype(str)
                # Заменяем 'nan' на специальное значение
                df[col] = df[col].replace('nan', 'MISSING')
                df[col] = df[col].replace('NaN', 'MISSING')
                df[col] = df[col].replace('<NA>', 'MISSING')
                
        return df
    
    def prepare_pools(self, data, stage='train'):
        """Подготовка данных в формате Pool для CatBoost"""
        
        if stage == 'train':
            # Убедимся, что категориальные признаки преобразованы
            x_tr_prepared = data['x_tr'].copy()
            x_val_prepared = data['x_val'].copy()
            
            self._prepare_categorical_features(x_tr_prepared, data.get('cat_features', []))
            self._prepare_categorical_features(x_val_prepared, data.get('cat_features', []))
            
            train_pool = Pool(
                data=x_tr_prepared,
                label=data['y_tr'],
                cat_features=self.cat_features_indices
            )
            
            val_pool = Pool(
                data=x_val_prepared,
                label=data['y_val'],
                cat_features=self.cat_features_indices
            )
            
            return train_pool, val_pool
            
        elif stage == 'final':
            # Убедимся, что категориальные признаки преобразованы
            x_train_prepared = data['x_train'].copy()
            x_oot_prepared = data['x_oot'].copy()
            
            self._prepare_categorical_features(x_train_prepared, data.get('cat_features', []))
            self._prepare_categorical_features(x_oot_prepared, data.get('cat_features', []))
            
            train_pool = Pool(
                data=x_train_prepared,
                label=data['y_train'],
                cat_features=self.cat_features_indices
            )
            
            oot_pool = Pool(
                data=x_oot_prepared,
                label=data['y_oot'],
                cat_features=self.cat_features_indices
            )
            
            return train_pool, oot_pool
    
    def tune(self, data):
        """Оптимизация гиперпараметров с Optuna для CatBoost"""
        
        # Подготовка данных в формате Pool
        train_pool, val_pool = self.prepare_pools(data, stage='train')
        
        # Создание study
        study = optuna.create_study(
            direction='maximize',
            study_name='catboost_optimization',
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        
        # Функция для оптимизации
        def objective_catboost(trial):
            params = {
                'iterations': trial.suggest_int('iterations', 100, 2000),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'depth': trial.suggest_int('depth', 4, 10),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
                'random_strength': trial.suggest_float('random_strength', 0.1, 10),
                'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1),
                'border_count': trial.suggest_int('border_count', 32, 255),
                'verbose': False,
                'random_seed': 42,
                'allow_writing_files': False,
                'task_type': 'CPU',
                'eval_metric': 'AUC',  # Явно указываем метрику
                'early_stopping_rounds': trial.suggest_int('early_stopping_rounds', 50, 200),
            }
            
            # Опциональные параметры
            if trial.suggest_categorical('use_rsm', [True, False]):
                params['rsm'] = trial.suggest_float('rsm', 0.1, 1.0)
            
            if trial.suggest_categorical('use_grow_policy', [True, False]):
                params['grow_policy'] = trial.suggest_categorical(
                    'grow_policy', ['SymmetricTree', 'Depthwise', 'Lossguide']
                )
            
            # Создание и обучение модели
            model = CatBoostClassifier(**params)
            
            model.fit(
                train_pool,
                eval_set=val_pool,
                verbose=False
            )
            
            # Получение лучшей метрики - несколько вариантов на случай изменения ключа
            best_score = model.get_best_score()
            
            # Пробуем разные возможные ключи
            if 'validation' in best_score and 'AUC' in best_score['validation']:
                return best_score['validation']['AUC']
            elif 'validation' in best_score and 'auc' in best_score['validation']:
                return best_score['validation']['auc']
            elif 'learn' in best_score and 'AUC' in best_score['learn']:
                return best_score['learn']['AUC']
            else:
                # Если не нашли AUC, ищем любую доступную метрику
                write_to_file(self.filepath, f"Доступные метрики: {best_score}")
                if best_score:
                    # Берем первую доступную метрику из validation
                    if 'validation' in best_score and best_score['validation']:
                        return list(best_score['validation'].values())[0]
                    else:
                        return list(best_score.values())[0]
                else:
                    # Вычисляем AUC вручную
                    from sklearn.metrics import roc_auc_score
                    y_pred_proba = model.predict_proba(val_pool)[:, 1]
                    y_val = val_pool.get_label()
                    return roc_auc_score(y_val, y_pred_proba)
        
        # Запуск оптимизации
        study.optimize(objective_catboost, n_trials=self.n_trials, show_progress_bar=True)
        
        # Сохранение лучших параметров
        self.best_params = study.best_params
        
        write_to_file(self.filepath, f"Лучший ROC-AUC: {study.best_value:.4f}")
        write_to_file(self.filepath, "Лучшие параметры CatBoost:")
        for key, value in self.best_params.items():
            write_to_file(self.filepath, f"  {key}: {value}")
        
        return self.best_params
    
    def fit(self, data, params):
        """Обучение финальной модели CatBoost"""
        
        # Подготовка данных
        train_pool, val_pool = self.prepare_pools(data, stage='final')
        
        # Базовые параметры CatBoost
        catboost_params = {
            'iterations': params.get('iterations', 1000),
            'learning_rate': params.get('learning_rate', 0.03),
            'depth': params.get('depth', 6),
            'l2_leaf_reg': params.get('l2_leaf_reg', 3),
            'random_strength': params.get('random_strength', 1),
            'bagging_temperature': params.get('bagging_temperature', 0.5),
            'border_count': params.get('border_count', 128),
            'verbose': 100,  # Вывод каждые 100 итераций
            'random_seed': 42,
            'allow_writing_files': False,
            'task_type': 'CPU',
            'early_stopping_rounds': params.get('early_stopping_rounds', 100),
            'use_best_model': True,
            'eval_metric': 'AUC',
        }
        
        # Добавление опциональных параметров
        optional_params = ['rsm', 'grow_policy']
        for param in optional_params:
            if param in params:
                catboost_params[param] = params[param]
        
        # Создание и обучение модели
        self.model = CatBoostClassifier(**catboost_params)
        
        self.model.fit(
            train_pool,
            eval_set=val_pool
        )
        
        write_to_file(self.filepath, f"Модель обучена. Количество итераций: {self.model.tree_count_}")
        write_to_file(self.filepath, f"Лучшая итерация: {self.model.get_best_iteration()}")
        write_to_file(self.filepath, f"Лучший AUC: {self.model.get_best_score()['validation']['AUC']:.4f}")
        
        return self.model
    
    def report(self, data):
        """Генерация отчета о качестве модели"""
        
        # Подготовка OOT данных
        _, oot_pool = self.prepare_pools(data, stage='final')
        
        # Предсказания
        y_pred_proba = self.model.predict_proba(oot_pool)[:, 1]
        
        # Генерация отчета
        report = metric_report(data['y_oot'], y_pred_proba)
        write_to_file(self.filepath, report)
        
        # Дополнительная информация
        write_to_file(self.filepath, "\nДополнительная информация CatBoost:")
        write_to_file(self.filepath, f"Размер OOT выборки: {len(data['y_oot'])}")
        write_to_file(self.filepath, f"Количество положительных классов в OOT: {sum(data['y_oot'])}")
        write_to_file(self.filepath, f"Количество деревьев: {self.model.tree_count_}")
        
        # Feature importance
        feature_importance = self.model.get_feature_importance()
        feature_names = list(data['x_oot'].columns)
        
        write_to_file(self.filepath, "\nТоп-10 важных признаков:")
        importance_pairs = sorted(zip(feature_names, feature_importance), 
                                 key=lambda x: x[1], reverse=True)[:10]
        for name, importance in importance_pairs:
            write_to_file(self.filepath, f"  {name}: {importance:.4f}")
        
        return report
    
    def run(self):
        """Запуск полного пайплайна"""
        try:
            # 1. Чтение данных
            df = self.read()
            
            # 2. Разделение и выбор фичей
            data = self.select(df)
            
            # 3. Оптимизация гиперпараметров
            best_params = self.tune(data)
            
            # 4. Обучение финальной модели
            model = self.fit(data, best_params)
            
            # 5. Формирование отчета
            report = self.report(data)
            
            return {
                'model': model,
                'params': best_params,
                'report': report,
                'filepath': self.filepath,
                'cat_features': self.cat_features_indices
            }
            
        except Exception as e:
            write_to_file(self.filepath, f"\nОШИБКА В ПАЙПЛАЙНЕ: {str(e)}")
            raise

# Парсинг аргументов
args = parse_args()
product_id = args.product_id
n_trials = args.n_trials
threshold = args.threshold

# Создание и запуск пайплайна
pipeline = CatBoostPipeline(product_id, n_trials, threshold)
results = pipeline.run()
  
print(f"Пайплайн CatBoost завершен. Результаты сохранены в: {results['filepath']}")
