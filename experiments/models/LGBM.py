import uuid
import kagglehub
import sys
import os
from datetime import datetime
from lightgbm import LGBMClassifier
import optuna
import category_encoders as ce

sys.path.append(os.path.join(os.path.dirname(__file__), '../..', 'utils'))

from target_constructor import target_constructor # type: ignore
from comp_db_dwld import process_incremental # type: ignore
from type_of_feature import type_of_feature # type: ignore
from metric_report import metric_report # type: ignore
from optunaizer_LGBM import objective # type: ignore
from out_file import write_to_file # type: ignore
from out_file import parse_args # type: ignore
from calib import calib_rep # type: ignore

class Pipeline:
    def __init__(self, product_id, n_trials, threshold):
        self.product_id = product_id
        self.n_trials = n_trials
        self.threshold = threshold
        self.filepath = f'../../artifacts/prod_{product_id}/LGBM_{datetime.now().strftime("%Y%m%d_%H%M%S")}_{str(uuid.uuid4())[:8]}.txt'
        
        self.encoder = None
        self.model = None
        self.best_params = None
        
    def read(self):        
        path = kagglehub.dataset_download("alexxl/sbol-dataset")
        df = process_incremental(path + '/sbolpro_merged_final.pqt', self.product_id)
        
        write_to_file(self.filepath, f"Размер исходных данных: {df.shape}")
        write_to_file(self.filepath, f"Колонки: {list(df.columns)}")
        
        return df
    
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
        
        # Определение типов фичей
        _, cat_cols = type_of_feature(df, self.threshold)
        cat_cols = cat_cols[:-5]  # Удаление последних 5 колонок
        num_cols = [col for col in x_train.columns if col not in cat_cols]
        
        write_to_file(self.filepath, f"Размер train: {x_train.shape}")
        write_to_file(self.filepath, f"Размер OOT: {x_oot.shape}")
        write_to_file(self.filepath, f"Категориальные фичи ({len(cat_cols)}): {cat_cols}")
        write_to_file(self.filepath, f"Числовые фичи ({len(num_cols)}): {num_cols}")
        
        return {
            'x_tr': x_tr, 'y_tr': y_tr,
            'x_val': x_val, 'y_val': y_val,
            'x_train': x_train, 'y_train': y_train,
            'x_oot': x_oot, 'y_oot': y_oot,
            'cat_cols': cat_cols, 'num_cols': num_cols
        }
    
    def encode(self, data, stage='train', smoothing=0.3):
        """Кодирование категориальных признаков"""
        
        cat_cols = data['cat_cols']
        num_cols = data['num_cols']

        if stage == 'train':
            # Создание и обучение энкодера
            self.encoder = ce.TargetEncoder(cols=cat_cols, smoothing=smoothing)
            
            # Кодирование train
            x_tr_enc = self.encoder.fit_transform(data['x_tr'], data['y_tr'])
            x_val_enc = self.encoder.transform(data['x_val'])
            # Восстановление числовых колонок
            x_tr_enc[num_cols] = data['x_tr'][num_cols]
            x_val_enc[num_cols] = data['x_val'][num_cols]
            
            write_to_file(self.filepath, f"Кодирование завершено. Smoothing: {smoothing}")
            
            return {
                'x_tr': x_tr_enc, 'y_tr': data['y_tr'],
                'x_val': x_val_enc, 'y_val': data['y_val']
            }
        
        elif stage == 'final':
            # Кодирование train
            x_tr_enc = self.encoder.fit_transform(data['x_tr'], data['y_tr'])
            x_val_enc = self.encoder.transform(data['x_val'])
            # Восстановление числовых колонок
            x_tr_enc[num_cols] = data['x_tr'][num_cols]
            x_val_enc[num_cols] = data['x_val'][num_cols]
            
            # Кодирование final train и OOT
            x_train_enc = self.encoder.fit_transform(data['x_train'], data['y_train'])
            x_oot_enc = self.encoder.transform(data['x_oot'])
            
            # Восстановление числовых колонок
            x_train_enc[num_cols] = data['x_train'][num_cols]
            x_oot_enc[num_cols] = data['x_oot'][num_cols]
            
            write_to_file(self.filepath, "Кодирование для финальной модели завершено")
            
            return {
                'x_train': x_train_enc, 'y_train': data['y_train'],
                'x_oot': x_oot_enc, 'y_oot': data['y_oot'],
                'x_val': x_val_enc, 'y_val': data['y_val']
            }
    
    def tune(self, data):
        """Оптимизация гиперпараметров с Optuna"""
        # Создание study
        study = optuna.create_study(
            direction='maximize',
            study_name='lgbm_optimization',
            sampler=optuna.samplers.TPESampler(seed=42)
        )

        # Запуск оптимизации
        study.optimize(
            lambda trial: objective(
                trial, 
                data['x_tr'], data['y_tr'], 
                data['x_val'], data['y_val'],
            ), 
            n_trials=self.n_trials, 
            show_progress_bar=True
        )

        # Сохранение лучших параметров
        self.best_params = study.best_params
        
        write_to_file(self.filepath, f"Лучший ROC-AUC: {study.best_value:.4f}")
        write_to_file(self.filepath, "Лучшие параметры:")
        for key, value in self.best_params.items():
            write_to_file(self.filepath, f"  {key}: {value}")
        
        return self.best_params

    def fit(self, data, params):
        """Обучение финальной модели"""
        
        # Создание модели с лучшими параметрами
        self.model = LGBMClassifier(
            n_estimators=params['n_estimators'],
            learning_rate=params['learning_rate'],
            num_leaves=params['num_leaves'],
            max_depth=params['max_depth'],
            min_child_samples=params['min_child_samples'],
            subsample=params['subsample'],
            colsample_bytree=params['colsample_bytree'],
            reg_alpha=params['reg_alpha'],
            reg_lambda=params['reg_lambda'],
            random_state=42,
            verbose=-1,
        )
        
        # Обучение модели
        self.model.fit(data['x_train'], data['y_train'])

        write_to_file(self.filepath, f"Модель обучена. Количество деревьев: {params['n_estimators']}")
        
        return self.model
    
    def report(self, data):
        """Генерация отчета о качестве модели"""
        
        # Предсказания
        y_pred_proba = self.model.predict_proba(data['x_oot'])[:, 1]
        y_pred_val = self.model.predict_proba(data['x_val'])[:, 1]

        prob_platt, prob_iso, prob_beta = calib_rep(y_pred_proba, data['y_oot'], y_pred_val, data['y_val'], self.product_id)
        
        # Генерация отчета
        report = metric_report(data['y_oot'], y_pred_proba)
        write_to_file(self.filepath, report)
        
        # Дополнительная информация
        write_to_file(self.filepath, "\nДополнительная информация:")
        write_to_file(self.filepath, f"Размер OOT выборки: {len(data['y_oot'])}")
        write_to_file(self.filepath, f"Количество положительных классов в OOT: {sum(data['y_oot'])}")
        
        return report
    
    def run(self):
        """Запуск полного пайплайна"""
        try:
            # 1. Чтение данных
            df = self.read()
            
            # 2. Разделение и выбор фичей
            data = self.select(df)
            
            # 3. Кодирование для тюнинга
            self.encode(data, stage='train', smoothing=0.3)
            # 4. Оптимизация гиперпараметров
            best_params = self.tune(data)  # Используем оригинальные данные для энкодера внутри objective
            
            # 5. Кодирование для финальной модели с лучшим smoothing
            encoded_data_final = self.encode(
                data, 
                stage='final', 
                smoothing=best_params.get('smoothing', 0.3)
            )
            # 6. Обучение финальной модели
            model = self.fit(encoded_data_final, best_params)
            
            # 7. Формирование отчета
            report = self.report(encoded_data_final)
            
            return {
                'model': model,
                'encoder': self.encoder,
                'params': best_params,
                'report': report,
                'filepath': self.filepath
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
pipeline = Pipeline(product_id, n_trials, threshold)
results = pipeline.run()
  
print(f"Пайплайн завершен. Результаты сохранены в: {results['filepath']}")