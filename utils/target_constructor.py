import pandas as pd

def target_constructor(df, product_id, drop_cols=['timestamp', 'user_id', 'product_id', 'year_month']):
    """
    Корректное разделение данных:
    - x_train, y_train: все кроме последнего месяца (для финальной модели)
    - x_tr, y_tr: train внутри TRAIN (все кроме последнего месяца TRAIN)
    - x_val, y_val: последний месяц TRAIN (для обучения калибратора)
    - x_oot, y_oot: последний месяц в данных (OOT)
    """
    # Фильтруем по продукту
    df_prod = df[df['product_id'] == product_id].copy()
    
    if df_prod.empty:
        raise ValueError(f"Продукт {product_id} не найден")

    # timestamp → datetime и period
    df_prod['timestamp'] = pd.to_datetime(df_prod['timestamp'])
    df_prod['period'] = df_prod['timestamp'].dt.to_period('M')
    
    # OOT = последний месяц
    last_period = df_prod['period'].max()
    mask_oot = df_prod['period'] == last_period
    df_oot = df_prod[mask_oot]
    
    # TRAIN = все кроме OOT
    mask_train_full = df_prod['period'] != last_period
    df_train_full = df_prod[mask_train_full]

    # Формируем x_train и y_train (все кроме последнего месяца)
    x_train = df_train_full.drop(columns=['is_sold', 'period'] + drop_cols)
    y_train = df_train_full['is_sold']

    # Сплит внутри TRAIN на tr/val
    last_train_month = df_train_full['period'].max()
    mask_val = df_train_full['period'] == last_train_month
    mask_tr = df_train_full['period'] != last_train_month

    df_val = df_train_full[mask_val]
    df_tr = df_train_full[mask_tr]

    x_tr = df_tr.drop(columns=['is_sold', 'period'] + drop_cols)
    y_tr = df_tr['is_sold']

    x_val = df_val.drop(columns=['is_sold', 'period'] + drop_cols)
    y_val = df_val['is_sold']

    x_oot = df_oot.drop(columns=['is_sold', 'period'] + drop_cols)
    y_oot = df_oot['is_sold']

    # Информация
    print("="*50)
    print(f"СТАТИСТИКА ПО ПРОДУКТУ: {product_id}")
    print("="*50)
    print(f"x_train (все кроме OOT): {x_train.shape[0]} строк")
    print(f"x_tr (train внутри TRAIN): {x_tr.shape[0]} строк")
    print(f"x_val (последний месяц TRAIN): {x_val.shape[0]} строк")
    print(f"x_oot (OOT): {x_oot.shape[0]} строк")
    print("-"*30)
    print(f"Процент продаж x_train: {y_train.mean()*100:.2f}%")
    print(f"Процент продаж x_tr:    {y_tr.mean()*100:.2f}%")
    print(f"Процент продаж x_val:   {y_val.mean()*100:.2f}%")
    print(f"Процент продаж x_oot:   {y_oot.mean()*100:.2f}%")
    print("="*50 + "\n")

    return x_tr, y_tr, x_val, y_val, x_train, y_train, x_oot, y_oot
