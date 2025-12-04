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
        

        #if i % 20 == 0:
        #    print(f"Обработано {(i+1)*batch_size:,} строк...")
    
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
        
        return dropped_features[-500:]
        
    except Exception as e:
        raise Exception(f"Ошибка при чтении файла {file_path}: {str(e)}")
