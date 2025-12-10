import pandas as pd
import pickle
import kagglehub

from comp_db_dwld import process_incremental # type: ignore
from out_file import parse_args # type: ignore
from target_constructor import target_constructor # type: ignore
from type_of_feature import type_of_feature # type: ignore

def predict_and_save(df_features: pd.DataFrame,
                                  model_path: str,
                                  calibrator_path: str,
                                  encoder_path: str,
                                  output_csv: str,
                                  categorical_cols: list):
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

    # 3. Загружаем энкодер
    with open(encoder_path, "rb") as f:
        encoder = pickle.load(f)

    # 4. Отделяем user_id для сохранения, но не используем в обучении/предсказании
    if "user_id" in df_features.columns:
        user_id = df_features["user_id"]
        df_encoded = df_features.drop(columns=["user_id"]).copy()
    else:
        user_id = [f"user_{i}" for i in range(len(df_features))]
        df_encoded = df_features.copy()

    # 5. Применяем энкодер к категориальным колонкам
    if categorical_cols:
        df_encoded[categorical_cols] = encoder.transform(df_encoded[categorical_cols])

    # 6. Предсказываем вероятности модели
    prob_raw = model.predict_proba(df_encoded)[:, 1]

    # 7. Применяем калибратор
    prob_calibrated = calibrator.predict_proba(df_encoded)[:, 1]

    # 8. Формируем итоговый DataFrame
    df_out = pd.DataFrame({
        "user_id": user_id,
        "chan_type": df_features["feature_969"],
        "score": prob_calibrated
    })

    # 9. Сохраняем в CSV
    df_out.to_csv(output_csv, index=False)
    print(f"Результат сохранён в {output_csv}")


args = parse_args()

model_path = f"./../artifacts/prod_{args.product_id}/LGBM/model.pkl"
output_csv = f"./../artifacts/prod_{args.product_id}/LGBM/scores.csv"
calibrator_path = f"./../artifacts/prod_{args.product_id}/LGBM/calibrator.pkl"
encoder_path = f"./../artifacts/prod_{args.product_id}/LGBM/encoder.pkl"

path = kagglehub.dataset_download("alexxl/sbol-dataset")
df = process_incremental(path + '/sbolpro_merged_final.pqt', args.product_id)
_, _, _, _, _, _, x_oot, _ = target_constructor(df, args.product_id)

x_oot = x_oot.reset_index(drop=True)
df = df.reset_index(drop=True)

x_oot["user_id"] = df.loc[x_oot.index, "user_id"]

_, cat_cols = type_of_feature(df, 20)
cat_cols = cat_cols[:-3]
num_cols = [col for col in x_oot.columns if col not in cat_cols]

predict_and_save(x_oot, model_path, calibrator_path, encoder_path, output_csv, cat_cols)