import pandas as pd
import numpy as np


def target_constructor(df, id_product):
    """
    Разделяет данные для конкретного product_id на Train (прошлые месяцы) и OOT (последний месяц).
    """
    # Фильтруем датасет по нужному продукту
    df_prod = df[df['product_id'] == id_product].copy()

    # Проверка: если данных по продукту нет
    if df_prod.empty:
        print(f"Ошибка: Продукт {id_product} не найден в датасете.")
        return None, None, None, None

    # Убеждаемся, что timestamp — это дата
    df_prod['timestamp'] = pd.to_datetime(df_prod['timestamp'])

    # Создаем вспомогательную колонку 'period' (год-месяц) для удобного поиска последнего месяца
    df_prod['period'] = df_prod['timestamp'].dt.to_period('M')

    # Находим последний месяц (максимальный период)
    last_period = df_prod['period'].max()

    # Разделяем на Train (всё кроме последнего) и OOT (только последний)
    mask_oot = df_prod['period'] == last_period

    df_train_full = df_prod[~mask_oot]  # Все прошлые месяцы
    df_oot_full = df_prod[mask_oot]  # Последний месяц

    # Формируем X и y
    # X - удаляем таргет (и вспомогательную колонку period)
    # y - только таргет
    x_train = df_train_full.drop(columns=['is_sold', 'period'])
    y_train = df_train_full['is_sold']

    x_oot = df_oot_full.drop(columns=['is_sold', 'period'])
    y_oot = df_oot_full['is_sold']

    # --- БЛОК ВЫВОДА ИНФОРМАЦИИ (PRINT) ---
    print("=" * 60)
    print(f"СТАТИСТИКА ПО ПРОДУКТУ: {id_product}")
    print("=" * 60)

    # Размеры выборок
    print(f"Размер x_train: {x_train.shape[0]} строк")
    print(f"Размер y_train: {y_train.shape[0]} строк")
    print(f"Размер x_oot:   {x_oot.shape[0]} строк")
    print(f"Размер y_oot:   {y_oot.shape[0]} строк")
    print("-" * 30)

    # Информация о периодах
    # Берем последний месяц из трейна (если трейн не пустой)
    if not df_train_full.empty:
        max_train_date = df_train_full['timestamp'].max()
        print(f"Последний месяц в TRAIN: {max_train_date.strftime('%B %Y')}")
    else:
        print("TRAIN пуст (возможно, у продукта данные только за 1 месяц)")

    # Месяц OOT
    print(f"Месяц OOT (Test):        {last_period.strftime('%B %Y')}")
    print("-" * 30)

    # Статистика по is_sold (конверсия) по месяцам
    print("Распределение is_sold (процент продаж) по месяцам:")

    # Объединяем обратно временно для красивого группирования, или используем исходный df_prod
    stats = df_prod.groupby('period')['is_sold'].agg(['count', 'mean'])
    stats['mean'] = (stats['mean'] * 100).round(2)
    stats.columns = ['Кол-во записей', 'Процент продаж %']

    # Помечаем, какая строка попала в OOT
    stats['Тип выборки'] = stats.index.map(lambda x: 'OOT' if x == last_period else 'TRAIN')

    print(stats)
    print("=" * 60 + "\n")

    return x_train, y_train, x_oot, y_oot