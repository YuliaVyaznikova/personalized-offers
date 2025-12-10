import pandas as pd
import numpy as np

def add_calc_columns_budget_only(df: pd.DataFrame, budget: float):
    df = df.copy()

    # prib = profit * prob / price
    df["prib_per_price"] = df["profit"] * df["prob"] / df["price"]
    df["prib"] = df["profit"] * df["prob"]

    # сортировка по прибыльности
    df = df.sort_values("prib_per_price", ascending=False).reset_index(drop=True)

    # подготовка колонок
    df["total_price"] = 0.0
    df["total_prib"] = 0.0
    df["itog"] = 0.0

    # создаём счётчики по типам
    types = df["type_comunication"].unique()
    for t in types:
        df[f"count_{t}"] = 0

    # внутренние переменные
    cumulative_price = 0.0
    cumulative_prib = 0.0
    type_counters = {t: 0 for t in types}

    # словарь для отслеживания использованных user_id по продуктам
    used_users = dict()  # {product_id: set(user_id)}

    # проходим по строкам
    for i, row in df.iterrows():
        t = row["type_comunication"]
        price = row["price"]
        prib = row["prib"]
        user = row["user_id"]
        product = row["product_id"]

        # инициализация множества для продукта
        if product not in used_users:
            used_users[product] = set()

        # проверяем, был ли уже user_id для этого продукта
        if user in used_users[product]:
            # пропускаем строку
            pass
        elif cumulative_price + price > budget:
            # превышение бюджета
            pass
        else:
            # строка подходит, учитываем
            cumulative_price += price
            cumulative_prib += prib
            type_counters[t] += 1
            used_users[product].add(user)

        # запись cumulative значений
        df.at[i, "total_price"] = cumulative_price
        df.at[i, "total_prib"] = cumulative_prib
        df.at[i, "itog"] = cumulative_prib / cumulative_price if cumulative_price > 0 else 0

        # запись счетчиков типа
        for tt in types:
            df.at[i, f"count_{tt}"] = type_counters[tt]

    # итоговый отчёт
    summary = {
        "spent_money": cumulative_price,
        "take_money": cumulative_prib,
        "profit": cumulative_prib - cumulative_price,
        "used_by_type": type_counters,
    }

    return summary

def add_calc_columns(df: pd.DataFrame, constraints: dict, budget: float):
    df = df.copy()

    # prib = profit * prob / price
    df["prib_per_price"] = df["profit"] * df["prob"] / df["price"]
    df["prib"] = df["profit"] * df["prob"]

    # сортировка по прибыльности
    df = df.sort_values("prib_per_price", ascending=False).reset_index(drop=True)

    # подготовка колонок
    df["total_price"] = 0.0
    df["total_prib"] = 0.0
    df["itog"] = 0.0

    # создаём счётчики по типам
    types = df["type_comunication"].unique()
    for t in types:
        df[f"count_{t}"] = 0

    # внутренние переменные
    cumulative_price = 0.0
    cumulative_prib = 0.0
    type_counters = {t: 0 for t in types}

    # словарь для отслеживания использованных user_id по продуктам
    used_users = dict()  # {product_id: set(user_id)}

    # проходим построчно
    for i, row in df.iterrows():
        t = row["type_comunication"]
        price = row["price"]
        prib = row["prib"]
        user = row["user_id"]
        product = row["product_id"]

        # инициализация множества для продукта
        if product not in used_users:
            used_users[product] = set()

        # проверка ограничений
        if type_counters[t] >= constraints.get(t, float("inf")):
            pass
        elif cumulative_price + price > budget:
            pass
        elif user in used_users[product]:
            pass
        else:
            # строка подходит
            type_counters[t] += 1
            cumulative_price += price
            cumulative_prib += prib
            used_users[product].add(user)

        # запись cumulative значений
        df.at[i, "total_price"] = cumulative_price
        df.at[i, "total_prib"] = cumulative_prib
        df.at[i, "itog"] = cumulative_prib / cumulative_price if cumulative_price > 0 else 0

        # запись счетчиков типа
        for tt in types:
            df.at[i, f"count_{tt}"] = type_counters[tt]

    # итоговый отчёт
    summary = {
        "spent_money": cumulative_price,
        "take_money": cumulative_prib,
        "profit": cumulative_prib - cumulative_price,
        "used_by_type": type_counters,
    }

    return summary

def add_product_id():
    """
    Самый простой вариант
    """
    # Сначала читаем prod_0
    df_full = pd.read_csv("./../artifacts/prod_0/LGBM/scores.csv")
    df_full['product_id'] = 0
    
    # Обратите внимание: range(1, 2, 4, 5, 6) не сработает, range() принимает только 1-3 аргумента
    # Лучше использовать список напрямую
    for i in [1, 2, 4, 5, 6]:
        # Убрал лишний pd.read_csv
        df = pd.read_csv(f"./../artifacts/prod_{i}/LGBM/scores.csv")
        df['product_id'] = i
        df_full = pd.concat([df_full, df], ignore_index=True)  # Важно: df_full должен быть первым
    
    df_full.to_csv("all.csv", index=False)
    print(f"✅ Готово! Сохранено {len(df_full)} строк")
add_product_id()

