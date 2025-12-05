import optuna
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score

def objective_catboost(trial, x_tr, y_tr, x_val, y_val, cat_features_indices):
    """
    Целевая функция для оптимизации CatBoost с Optuna
    """
    # Подготовка данных в формате Pool
    train_pool = Pool(data=x_tr, label=y_tr, cat_features=cat_features_indices)
    val_pool = Pool(data=x_val, label=y_val, cat_features=cat_features_indices)
    
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
    
    # Получение предсказаний
    y_pred_proba = model.predict_proba(val_pool)[:, 1]
    
    # Расчет метрики
    auc_score = roc_auc_score(y_val, y_pred_proba)
    
    return auc_score