import pandas as pd
import pickle
import kagglehub

from comp_db_dwld import process_incremental, process_last_month_all_products # type: ignore
from target_constructor import target_constructor # type: ignore

def predict_and_save(df_features: pd.DataFrame,
                                  model_path: str,
                                  calibrator_path: str,
                                  output_csv: str,):
    """
    df_features: DataFrame с колонкой 'feature_969', 'user_id' + остальные фичи
    model_path: путь к .pkl модели
    calibrator_path: путь к .pkl калибратора
    encoder_path: путь к .pkl энкодера категориальных признаков
    output_csv: путь для сохранения CSV
    categorical_cols: список категориальных колонок для энкодера
    """
    # 1. Загружаем модель
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # 2. Загружаем калибратор
    with open(calibrator_path, "rb") as f:
        calibrator = pickle.load(f)

    # 4. Отделяем user_id для сохранения, но не используем в обучении/предсказании
    if "user_id" in df_features.columns:
        user_id = df_features["user_id"]
        df_encoded = df_features.drop(columns=["user_id"]).copy()
    else:
        user_id = [f"user_{i}" for i in range(len(df_features))]
        df_encoded = df_features.copy()

    # 6. Предсказываем вероятности модели
    prob_raw = model.predict_proba(df_encoded)[:, 1]

    # 7. Применяем калибратор
    prob_calibrated = calibrator.predict(prob_raw)

    # 8. Формируем итоговый DataFrame
    df_out = pd.DataFrame({
        "user_id": user_id,
        "score": prob_calibrated
    })

    # 9. Сохраняем в CSV
    df_out.to_csv(output_csv, index=False)
    print(f"Результат сохранён в {output_csv}")


for i in [0, 1, 2, 4, 5, 6]:
    model_path = f"./../artifacts/prod_{i}/LGBM/model.pkl"
    output_csv = f"./../artifacts/prod_{i}/LGBM/scores.csv"
    calibrator_path = f"./../artifacts/prod_{i}/LGBM/calibrator.pkl"
    encoder_path = f"./../artifacts/prod_{i}/LGBM/encoder.pkl"

    path = kagglehub.dataset_download("alexxl/sbol-dataset")

    df = process_incremental(path + '/sbolpro_merged_final.pqt', i)
    _, _, _, _, _, _, x_oot_all, _ = target_constructor(df, i)

    with open(encoder_path, "rb") as f:
        encoder = pickle.load(f)
    x_oot_all = encoder.transform(x_oot_all)
    x_oot_all = x_oot_all.reset_index(drop=True)
    df = df.reset_index(drop=True)

    x_oot_all["user_id"] = df.loc[x_oot_all.index, "user_id"]

    df = process_last_month_all_products(path + '/sbolpro_merged_final.pqt', x_oot_all.columns)
    predict_and_save(df, model_path, calibrator_path, output_csv)