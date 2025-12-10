import pandas as pd
import gc
import pyarrow.parquet as pq
import os

# Метод 3: Итеративная обработка с немедленным удалением
def process_incremental(file_path, product_id):
    """Постепенная обработка с немедленным удалением ненужных данных"""
    print("\n🔄 Постепенная обработка...")

    required_columns = read_rc(product_id)
    
    # Читаем файл по частям
    batch_size = 10000
    result_chunks = []
    
    # Создаем ParquetFile объект
    pq_file = pq.ParquetFile(file_path)


    required_columns += ['timestamp', 'user_id', 'product_id', 'is_sold']

    for i, batch in enumerate(pq_file.iter_batches(batch_size=batch_size, columns=required_columns)):
        df_batch = batch.to_pandas()
        
        # Немедленная фильтрация
        df_batch['timestamp'] = pd.to_datetime(df_batch['timestamp'])
        df_batch['year_month'] = df_batch['timestamp'].dt.strftime('%Y-%m')
        
        # Удаляем май 2023
        df_batch = df_batch[df_batch['year_month'] != '2023-05']
        
        if len(df_batch) > 0:
            # Оптимизируем типы
            for col in df_batch.select_dtypes(include=['float64']).columns:
                df_batch[col] = df_batch[col].astype('float32')
            
            result_chunks.append(df_batch)
        
        # Очистка памяти после каждого батча
        del df_batch
        gc.collect()
            
    # Объединяем результаты
    if result_chunks:
        result = pd.concat(result_chunks, ignore_index=True)
        #print(f"✅ Итоговый размер: {len(result):,} строк")
        return result
    else:
        print("⚠️  Нет данных после фильтрации")
        return pd.DataFrame()
    
import os
import pandas as pd

def read_rc(id_product):
    """Читает список колонок из колонки dropped_feature файла rfe_metrics_history_{id_product}"""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Формируем путь к файлу с историей RFE
    file_path = os.path.join(
        project_root, 
        "utils", 
        "best_feature", 
        f"rfe_metrics_history_{id_product}.csv"
    )
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Файл не найден: {file_path}")
    
    try:
        df = pd.read_csv(file_path)
        if 'dropped_feature' not in df.columns:
            raise ValueError(f"Колонка 'dropped_feature' не найдена в файле {file_path}")
        
        dropped_features = df['dropped_feature'].dropna().astype(str).tolist()
        
        return dropped_features[-200:]
        
    except Exception as e:
        raise Exception(f"Ошибка при чтении файла {file_path}: {str(e)}")


import pandas as pd
import pyarrow.parquet as pq
import gc

def process_last_month_all_products(file_path, use_columns):
    """
    Возвращает строки последнего месяца (OOT) по каждому product_id,
    при этом читает и возвращает только указанные колонки use_columns.

    Пример:
        use_columns = ['user_id', 'product_id', 'timestamp', 'feature_101']
    """

    # timestamp и product_id нужны всегда для определения последнего месяца
    required_helper_cols = ['timestamp', 'product_id']
    read_cols = list(set(list(use_columns) + required_helper_cols))

    # ---------- Первый проход: определяем последний месяц ----------
    pq_file = pq.ParquetFile(file_path)
    last_periods = {}

    for batch in pq_file.iter_batches(batch_size=50000, columns=read_cols):
        df = batch.to_pandas()

        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['year_month'] = df['timestamp'].dt.strftime('%Y-%m')

        #  Удаляем май 2023
        df = df[df['year_month'] != '2023-05']
        if df.empty:
            continue

        df['period'] = df['timestamp'].dt.to_period('M')

        # Определяем последний месяц для каждого product_id
        grouped = df.groupby('product_id')['period'].max()
        for pid, period in grouped.items():
            if pid not in last_periods:
                last_periods[pid] = period
            else:
                last_periods[pid] = max(last_periods[pid], period)

        del df, grouped
        gc.collect()

    # ---------- Второй проход: собираем только последнюю дату ----------
    pq_file = pq.ParquetFile(file_path)
    result_batches = []

    for batch in pq_file.iter_batches(batch_size=50000, columns=read_cols):
        df = batch.to_pandas()

        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['year_month'] = df['timestamp'].dt.strftime('%Y-%m')
        #  Удаляем май 2023
        df = df[df['year_month'] != '2023-05']
        if df.empty:
            continue

        df['period'] = df['timestamp'].dt.to_period('M')

        # Маска OOT
        mask = df.apply(
            lambda r: r['product_id'] in last_periods 
                      and r['period'] == last_periods[r['product_id']],
            axis=1
        )

        df_oot = df[mask]

        if not df_oot.empty:
            # возвращаем только user-выбранные столбцы
            df_oot = df_oot[use_columns]
            result_batches.append(df_oot)

        del df, df_oot, mask
        gc.collect()

    if not result_batches:
        print("⚠️ Нет OOT данных!")
        return pd.DataFrame(columns=use_columns)

    df_final = pd.concat(result_batches, ignore_index=True)

    print(f"\n🔥 Итоговый размер результата: {len(df_final):,} строк")
    return df_final
