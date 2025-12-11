import pandas as pd
from typing import Dict, List

def add_product_id():
    """
    Самый простой вариант
    """
    # Сначала читаем prod_0
    df_full = pd.read_csv("./../artifacts/prod_0/LGBM/scores.csv")
    df_full['product_id'] = 0
    
    for i in [1, 2, 4, 5, 6]:
        # Убрал лишний pd.read_csv
        df = pd.read_csv(f"./../artifacts/prod_{i}/LGBM/scores.csv")
        df['product_id'] = i
        df_full = pd.concat([df_full, df], ignore_index=True)  # Важно: df_full должен быть первым
    
    df_full.to_csv("all.csv", index=False)
    print(f"✅ Готово! Сохранено {len(df_full)} строк")

def compute_weighted_scores(price_map: Dict[int, float]) -> List[float]:
    # Загружаем CSV
    df = pd.read_csv("all.csv")

    # Применяем price_map
    df['price'] = df['product_id'].map(price_map)

    # Вычисляем price * score
    df['weighted_score'] = df['price'] * df['score']

    return df['weighted_score'].tolist()


def knapsack_products_with_limits(product_options, budget, comm_limits):
    n = len(product_options)
    num_comms = len(comm_limits)
    
    prev = {(0, tuple([0]*num_comms)): (0, [])}

    for i in range(n):
        curr = {}
        options = product_options[i]
        
        for (spent, usage), (profit, selected) in prev.items():
            for variant_idx, (cost, new_dohod, comm_idx) in enumerate(options):
                new_spent = spent + cost
                if new_spent > budget:
                    continue
                
                usage_list = list(usage)
                if comm_idx > 0:
                    usage_list[comm_idx-1] += 1
                    if usage_list[comm_idx-1] > comm_limits[comm_idx-1]:
                        continue
                
                new_usage = tuple(usage_list)
                new_profit = profit + new_dohod
                new_selected = selected + [variant_idx]
                
                key = (new_spent, new_usage)
                if key not in curr or curr[key][0] < new_profit:
                    curr[key] = (new_profit, new_selected)
        
        prev = curr

    best_profit, best_selected = max(prev.values(), key=lambda x: x[0])
    return best_profit, best_selected


def build_product_options(weighted_scores, communication_options):
    """
    Для каждого продукта формирует список вариантов (price, new_dohod), 
    где каждый вариант соответствует применению одной из коммуникаций.

    :param weighted_scores: список весов продукта (weighted_score)
    :param communication_options: список кортежей (cost, prob, limit) для каждой коммуникации
    :return: список списков [(price, new_dohod), ...] для каждого продукта
    """
    product_options = []

    for s in weighted_scores:
        options_for_product = [(0, 0)]  # вариант "не применять коммуникацию"
        
        for cost, prob, _limit in communication_options:
            new_dohod = s * prob
            options_for_product.append((cost, new_dohod))
        
        product_options.append(options_for_product)

    return product_options


price_map = {
    0: 15807.9,
    1: 21368,
    2: 256000,
    6: 1000,
    4: 20000,
    5: 90000,
}
weighted_scores = compute_weighted_scores(price_map)

communication_options = [
    (0, 0, 0),        # вариант "не использовать"
    (35, 0.02, 1),     # (стоимость, вероятность, comm_idx)
    (2, 0.03, 2),
    (5, 0.015, 3),
    (1500, 0.07, 4),
    (120, 0.01, 5),
    (80, 0.005, 6),

]

product_options = build_product_options(weighted_scores, communication_options)

product_options = []
for s in weighted_scores:
    opts = []
    for cost, prob, comm_idx in communication_options:
        new_dohod = s * prob
        opts.append((cost, new_dohod, comm_idx))
    product_options.append(opts)

budget = 1000000
comm_limits = [2000, 5000, 1200, 80, 600, 900]  # первая коммуникация можно использовать 1 раз, вторая — 2 раза

best_profit, best_variants = knapsack_products_with_limits(product_options, budget, comm_limits)
print("Максимальная прибыль:", best_profit)
print("Выбранные варианты для каждого продукта:", best_variants)