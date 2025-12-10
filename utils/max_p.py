import pandas as pd
import numpy as np

def add_calc_columns_budget_only_unique_users(df: pd.DataFrame, budget: float):
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

def add_calc_columns_unique_users(df: pd.DataFrame, constraints: dict, budget: float):
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


# --------------------------
# Генерация DataFrame
# --------------------------
np.random.seed(42)  # для воспроизводимости

n_rows = 15311
types = ["email", "sms", "call"]

# фиксированные profit и price для каждого типа
profit_dict = {"email": 10000, "sms": 15000, "call": 30000}
price_dict  = {"email": 50, "sms": 60, "call": 120}

# случайный выбор типа для каждой строки
type_com = np.random.choice(types, size=n_rows)

# profit и price в зависимости от типа
profit = [profit_dict[t] for t in type_com]
price  = [price_dict[t] for t in type_com]

# случайная вероятность успеха
prob = np.random.uniform(0.1, 1.0, size=n_rows)

# создаём DataFrame
df = pd.DataFrame({
    "type_comunication": type_com,
    "profit": profit,
    "price": price,
    "prob": prob
})

# --------------------------
# Ограничения и бюджет
# --------------------------
constraints = {"email": 50000, "sms": 400000, "call": 3000}
budget = 1_000_000  # например 1 млн
print(add_calc_columns_budget_only(df, budget))
print(add_calc_columns(df, constraints, budget))