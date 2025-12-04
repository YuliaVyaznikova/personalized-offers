from sklearn.metrics import roc_auc_score
import category_encoders as ce
from lightgbm import LGBMClassifier

def objective(trial, x_tr, y_tr, x_val, y_val, cat_cols, num_cols):              
    # 2. smoothing для TargetEncoder
    smoothing = trial.suggest_float('smoothing', 0.1, 0.9, step=0.1)
    # 3. Параметры LightGBM
    n_estimators = trial.suggest_int('n_estimators', 50, 500, step=50)
    learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
    num_leaves = trial.suggest_int('num_leaves', 20, 150, step=10)
    max_depth = trial.suggest_int('max_depth', 3, 12)
    min_child_samples = trial.suggest_int('min_child_samples', 10, 100, step=10)
    subsample = trial.suggest_float('subsample', 0.6, 1.0, step=0.1)
    colsample_bytree = trial.suggest_float('colsample_bytree', 0.6, 1.0, step=0.1)
    reg_alpha = trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True)
    reg_lambda = trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True)
    
    # Преобразуем категориальные признаки с оптимизируемым smoothing
    target_encoder = ce.TargetEncoder(cols=cat_cols, smoothing=smoothing)
    x_tr_enc = target_encoder.fit_transform(x_tr, y_tr)
    x_val_enc = target_encoder.transform(x_val)
    
    # Оставляем числовые
    x_tr_enc[num_cols] = x_tr[num_cols]
    x_val_enc[num_cols] = x_val[num_cols]
    
    # Конфигурация модели с оптимизируемыми параметрами
    model = LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        max_depth=max_depth,
        min_child_samples=min_child_samples,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        random_state=42,
        verbose=-1
    )
    
    # Обучение модели
    model.fit(x_tr_enc, y_tr)
    
    # Предсказание на валидации
    y_val_pred = model.predict_proba(x_val_enc)[:, 1]
    
    # ROC-AUC как метрика для оптимизации
    score = roc_auc_score(y_val, y_val_pred)
    
    return score