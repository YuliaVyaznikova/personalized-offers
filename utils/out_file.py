import argparse
import json

def write_to_file(filename, content, mode='a'):
    # Если content - словарь, преобразуем его в читаемую строку
    if isinstance(content, dict):
        # Форматируем словарь как JSON с отступами для читаемости
        content = json.dumps(content, indent=2, ensure_ascii=False)
    
    with open(filename, mode, encoding='utf-8') as f:
        f.write(content)
        f.write("\n")

def parse_args():
    parser = argparse.ArgumentParser(description='Обучение LGBM модели')
    parser.add_argument('--product-id', type=int, required=True,
                       help='ID продукта для обучения модели')
    parser.add_argument('--n-trials', type=int, default=30,
                       help='Количество trials для Optuna (по умолчанию: 30)')
    parser.add_argument('--threshold', type=int, default=20,
                       help='Порог для определения категориальных признаков (по умолчанию: 20)')
    return parser.parse_args()