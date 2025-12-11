import json
import pandas as pd
import numpy as np
import os
from mip import Model, xsum, BINARY, MAXIMIZE, OptimizationStatus
import random


# ==========================================
# 1. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ==========================================

def load_and_prepare_data(input_json_path):
    """
    Читает JSON конфиг и соответствующие CSV файлы с вероятностями.
    Возвращает подготовленные структуры данных для оптимизатора.
    """
    # 1. Читаем конфиг
    with open(input_json_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    budget = config['budget']

    # Парсим каналы
    # JSON формат: "sms": [limit, cost, prob]
    channels_data = {}
    channel_limits = {}

    for c_name, params in config['channels'].items():
        limit, cost, prob = params
        channels_data[c_name] = {'cost': cost, 'prob': prob}
        channel_limits[c_name] = int(limit)

    # Парсим продукты и собираем данные из CSV
    product_revenues = config['products']

    affinity_dfs = []

    # Читаем CSV для каждого продукта
    # Ожидаем, что файл называется {product_id}.csv и лежит рядом
    # Формат CSV: client_id, probability
    """
    for product_id, revenue in product_revenues.items():
        csv_name = f"df_0_filtered.csv"

        if os.path.exists(csv_name):
            # Читаем без заголовка или с заголовком, здесь предполагаем наличие заголовка
            # Если заголовка нет, используйте header=None и names=['client_id', 'prob']
            try:
                df = pd.read_csv(csv_name)
                # Нормализуем имена колонок для надежности
                df.columns = ['client_id', 'affinity_prob']
                df['product_id'] = product_id
                affinity_dfs.append(df)
            except Exception as e:
                print(f"Ошибка чтения {csv_name}: {e}")
        else:
            print(f"ВНИМАНИЕ: Файл {csv_name} не найден. Продукт пропущен.")

    if not affinity_dfs:
        raise ValueError("Не найдено ни одного файла с данными по продуктам!")
"""
    # Объединяем все в один большой DataFrame


    df_affinity = pd.read_csv("df_0_filtered.csv")
    product_choices = list(product_revenues.keys())
    df_affinity['product_id'] = np.random.choice(product_choices, size=len(df_affinity))

    df_affinity.to_csv("test.csv", index=False)


    # Список уникальных клиентов
    clients = df_affinity['user_id'].unique().tolist()

    return clients, product_revenues, channels_data, channel_limits, df_affinity, budget


def solve_mip_model(clients, product_revenues, channels_data, channel_limits, df_affinity, budget):
    """
    Строит и решает MIP модель.
    """
    print("Построение математической модели...")
    model = Model(sense=MAXIMIZE, solver_name='CBC')
    model.verbose = 0

    # --- Переменные и Коэффициенты ---
    # x[(client, product, channel)] -> binary variable
    x = {}
    coeffs = {}  # Храним тут revenue и cost для быстрого доступа

    # Чтобы ускорить создание переменных, итерируемся по DataFrame предрасположенности
    # так как это уже разреженная матрица (client-product)

    # Преобразуем данные каналов в списки для быстрого перебора
    channel_names = list(channels_data.keys())

    for row in df_affinity.itertuples(index=False):
        client = row.user_id
        product = row.product_id
        p_affinity = row.lama_proba

        if p_affinity <= 0: continue

        for channel in channel_names:
            c_data = channels_data[channel]

            # Создаем переменную решения
            var_key = (client, product, channel)
            x[var_key] = model.add_var(var_type=BINARY)

            # Считаем ожидаемую выручку и стоимость
            # Revenue = Price * P(Client) * P(Channel)
            exp_rev = product_revenues[product] * p_affinity * c_data['prob']
            cost = c_data['cost']

            coeffs[var_key] = {'rev': exp_rev, 'cost': cost}

    # --- Целевая функция (Максимизация выручки) ---
    model.objective = xsum(x[k] * coeffs[k]['rev'] for k in x)

    # --- Ограничение 1: Бюджет ---
    model += xsum(x[k] * coeffs[k]['cost'] for k in x) <= budget

    # --- Ограничение 2: 1 клиент = макс 1 предложение ---
    # Группируем переменные по клиентам
    # (В продакшене для скорости можно использовать defaultdict при создании x)
    client_to_vars = {}
    for key, var in x.items():
        client = key[0]
        if client not in client_to_vars:
            client_to_vars[client] = []
        client_to_vars[client].append(var)

    for client, vars_list in client_to_vars.items():
        model += xsum(vars_list) <= 1

    # --- Ограничение 3: Лимиты каналов ---
    # Группируем переменные по каналам
    channel_to_vars = {ch: [] for ch in channel_names}
    for key, var in x.items():
        channel_to_vars[key[2]].append(var)

    for channel, limit in channel_limits.items():
        if channel in channel_to_vars and channel_to_vars[channel]:
            model += xsum(channel_to_vars[channel]) <= limit

    # --- Решение ---
    print("Запуск солвера...")
    model.optimize()

    # --- Сбор результатов ---
    results = []
    if model.status in [OptimizationStatus.OPTIMAL, OptimizationStatus.FEASIBLE]:
        print(f"Решение найдено! Целевое значение: {model.objective_value:.2f}")
        for key, var in x.items():
            if var.x >= 0.99:
                client, product, channel = key
                results.append({
                    'product_id': product,
                    'client_id': client,
                    'canal_id': channel,
                    'cost': coeffs[key]['cost'],
                    'expected_revenue': coeffs[key]['rev']
                })
    else:
        print("Оптимальное решение не найдено.")

    return pd.DataFrame(results)


def generate_reports(df_results, budget, product_revenues, channels_data, channel_limits, output_json_path,
                     output_csv_path):
    """
    Формирует и сохраняет отчеты.
    """
    # 1. Сохраняем CSV
    # Формат: product_id - client_id - canal_id
    out_columns = ['product_id', 'client_id', 'canal_id']
    if not df_results.empty:
        df_results[out_columns].to_csv(output_csv_path, index=False)
    else:
        # Создаем пустой файл с заголовками, если решений нет
        pd.DataFrame(columns=out_columns).to_csv(output_csv_path, index=False)

    # 2. Формируем JSON
    if df_results.empty:
        report = {
            "summary": [budget, 0, 0.0, 0, 0.0, 0],
            "channels_usage": {},
            "products_distribution": {}
        }
    else:
        # --- Summary ---
        actual_spend = df_results['cost'].sum()
        actual_spend_percent = round((actual_spend / budget) * 100, 1)
        expected_revenue = df_results['expected_revenue'].sum()
        expected_roi = ((expected_revenue - actual_spend) / actual_spend * 100) if actual_spend > 0 else 0
        reach_clients = len(df_results)

        # summary: [budget_available, actual_spend, actual_spend_percent, expected_revenue, expected_roi_percent, reach_clients]
        summary_vec = [
            budget,
            round(actual_spend, 2),
            actual_spend_percent,
            round(expected_revenue, 2),
            round(expected_roi, 1),
            reach_clients
        ]

        # --- Channels Usage ---
        # channel -> [limit, offers_count, total_cost, total_revenue]
        # (в примере было 3 числа, но добавим лимит для ясности, или строго как в примере: [limit, actual_count, revenue_from_channel?])
        # В примере: "sms": [200000 (limit), 70000 (count), 140000 (revenue? or cost?)]
        # В примере sms cost = 0.35. 70000 * 0.35 = 24500. А написано 140000. Скорее всего 3-е число это Revenue.
        # Давайте делать: [limit, count, revenue]

        channels_usage = {}
        for ch in channels_data.keys():
            ch_df = df_results[df_results['canal_id'] == ch]
            count = len(ch_df)
            rev = round(ch_df['expected_revenue'].sum(), 2)
            limit = channel_limits.get(ch, 0)
            channels_usage[ch] = [limit, count, rev]

        # --- Products Distribution ---
        # product -> [offers_count, revenue_per_unit] (как в примере)
        # Или [offers_count, total_revenue]?
        # В примере: "debit_card": [7150, 1100]. 1100 это цена одной карты.
        products_distribution = {}
        for prod in product_revenues.keys():
            prod_df = df_results[df_results['product_id'] == prod]
            count = len(prod_df)
            # В примере вторым числом идет константа заработка с продукта
            products_distribution[prod] = [count, product_revenues[prod]]

        report = {
            "summary": summary_vec,
            "channels_usage": channels_usage,
            "products_distribution": products_distribution
        }

    # Сохраняем JSON
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=4, ensure_ascii=False)

    return output_json_path


# ==========================================
# 2. ОСНОВНАЯ ФУНКЦИЯ
# ==========================================

def optimize_marketing_campaign(input_json_path):
    """
    Основная функция-оркестратор.
    1. Читает настройки и данные.
    2. Запускает оптимизацию.
    3. Генерирует файлы отчетов.

    Returns:
        str: Имя созданного JSON файла с отчетом.
    """
    output_json = 'report.json'
    output_csv = 'results.csv'

    # 1. Загрузка
    try:
        clients, prod_revs, chan_data, chan_limits, df_aff, budget = load_and_prepare_data(input_json_path)
    except Exception as e:
        print(f"Ошибка при загрузке данных: {e}")
        return None

    # 2. Оптимизация
    df_results = solve_mip_model(
        clients,
        prod_revs,
        chan_data,
        chan_limits,
        df_aff,
        budget
    )

    # 3. Отчетность
    report_file = generate_reports(
        df_results,
        budget,
        prod_revs,
        chan_data,
        chan_limits,
        output_json,
        output_csv
    )

    print(f"Готово. Отчет сохранен в {report_file}, детализация в {output_csv}")
    return report_file


# ==========================================
# 3. ТЕСТИРОВАНИЕ (Генерация данных и запуск)
# ==========================================

def create_dummy_environment():
    """
    Создает input.json для проверки работы скрипта.
    """
    print("Генерация тестовых данных...")

    # 1. Создаем input.json
    config = {
        "model": "test_model",
        "budget": random.randint(5000, 400000),  # Небольшой бюджет, чтобы алгоритм выбирал
        "advanced": True,
        "enable_rr": True,
        "channels": {
            "sms": [random.randint(5000, 50000), random.uniform(0.01, 0.1), random.uniform(0.001, 0.00001)],  # [limit, cost, prob]
            "email": [random.randint(5000, 50000), random.uniform(0.002, 0.01), random.uniform(0.001, 0.0001)],
            "push": [random.randint(10000, 5000000), random.uniform(0.001, 0.01), random.uniform(0.001, 0.00001)],
            "phone": [random.randint(5000, 10000), random.uniform(4, 2), random.uniform(0.01, 0.0001)]  # Очень дорогой и лимитированный канал
        },
        "products": {
            "credit_card": random.randint(5000, 400000),
            "debit_card": random.randint(1000, 10000),
            "loan": random.randint(5000, 400000), 
            "auto_c": random.randint(50000, 400000),
            "subscr" : random.uniform(100, 1000),
            "cash": random.uniform(10000, 400000),
        }
    }

    with open('input.json', 'w') as f:
        json.dump(config, f, indent=2)

    print("Тестовые файлы созданы: input.json")


# --- ЗАПУСК ТЕСТА ---

if __name__ == "__main__":
    # 1. Создаем окружение
    create_dummy_environment()

    # 2. Запускаем функцию
    result_file = optimize_marketing_campaign('input.json')

    # 3. Читаем и выводим результат для проверки
    if result_file:
        print("\n--- Содержимое отчета (JSON) ---")
        with open(result_file, 'r') as f:
            print(json.dumps(json.load(f), indent=2))

        print("\n--- Пример CSV (первые 5 строк) ---")
        try:
            print(pd.read_csv('results.csv').head())
        except:
            print("Файл results.csv пуст или не создан.")