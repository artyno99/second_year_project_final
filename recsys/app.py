"""Веб-сервис рекомендаций.

Сервис читает заранее рассчитанные рекомендации (artifacts/recommendations.npz)
и выдаёт три товара по идентификатору пользователя. Пользователям без истории
возвращаются популярные товары.

Локальный запуск:  python app.py
В контейнере:      gunicorn --bind 0.0.0.0:8000 --workers 1 --threads 8 app:app
"""
import json
import os
import threading
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from flask import Flask, Response, g, jsonify, request
from werkzeug.exceptions import HTTPException

ARTIFACTS_DIR = Path(os.environ.get('ARTIFACTS_DIR', Path(__file__).parent / 'artifacts'))
MAX_K = 3                    # сколько рекомендаций рассчитано для каждого пользователя
MAX_BATCH = 1000             # максимум пользователей в одном пакетном запросе
MAX_USER_ID = 2 ** 63 - 1


class Recommender:
    """Хранилище готовых рекомендаций с быстрым поиском по идентификатору пользователя."""

    def __init__(self, artifacts_dir):
        data = np.load(artifacts_dir / 'recommendations.npz')
        self.user_ids = data['user_ids']            # отсортированный массив идентификаторов
        self.items = data['items']                  # матрица (пользователи x 3)
        self.popular = [int(i) for i in data['popular']]
        with open(artifacts_dir / 'model_info.json', encoding='utf-8') as f:
            self.info = json.load(f)

    def recommend(self, user_id, k=MAX_K):
        """Возвращает (список из k товаров, признак персональной рекомендации)."""
        pos = int(np.searchsorted(self.user_ids, user_id))
        if pos < len(self.user_ids) and self.user_ids[pos] == user_id:
            return [int(i) for i in self.items[pos][:k]], True
        return self.popular[:k], False


class Metrics:
    """Счётчики сервиса в формате Prometheus (без внешних зависимостей)."""

    BUCKETS = (0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0)

    def __init__(self):
        self.lock = threading.Lock()
        self.started = time.time()
        self.requests = defaultdict(int)             # (endpoint, method, status) -> число запросов
        self.latency_sum = defaultdict(float)        # endpoint -> суммарное время, с
        self.latency_count = defaultdict(int)
        self.latency_buckets = defaultdict(lambda: [0] * len(self.BUCKETS))
        self.recommendations = defaultdict(int)      # personalized / popular -> число выданных рекомендаций

    def observe_request(self, endpoint, method, status, seconds):
        with self.lock:
            self.requests[(endpoint, method, status)] += 1
            self.latency_sum[endpoint] += seconds
            self.latency_count[endpoint] += 1
            for i, bound in enumerate(self.BUCKETS):
                if seconds <= bound:
                    self.latency_buckets[endpoint][i] += 1

    def observe_recommendation(self, personalized):
        with self.lock:
            self.recommendations['personalized' if personalized else 'popular'] += 1

    def render(self, recommender):
        info = recommender.info
        lines = [
            '# HELP recsys_requests_total Число HTTP-запросов.',
            '# TYPE recsys_requests_total counter',
        ]
        with self.lock:
            for (endpoint, method, status), value in sorted(self.requests.items()):
                lines.append(f'recsys_requests_total{{endpoint="{endpoint}",method="{method}",status="{status}"}} {value}')
            lines += ['# HELP recsys_request_duration_seconds Время обработки запроса.',
                      '# TYPE recsys_request_duration_seconds histogram']
            for endpoint in sorted(self.latency_count):
                for bound, value in zip(self.BUCKETS, self.latency_buckets[endpoint]):
                    lines.append(f'recsys_request_duration_seconds_bucket{{endpoint="{endpoint}",le="{bound}"}} {value}')
                lines.append(f'recsys_request_duration_seconds_bucket{{endpoint="{endpoint}",le="+Inf"}} '
                             f'{self.latency_count[endpoint]}')
                lines.append(f'recsys_request_duration_seconds_sum{{endpoint="{endpoint}"}} {self.latency_sum[endpoint]:.6f}')
                lines.append(f'recsys_request_duration_seconds_count{{endpoint="{endpoint}"}} {self.latency_count[endpoint]}')
            lines += ['# HELP recsys_recommendations_total Число выданных рекомендаций по типам.',
                      '# TYPE recsys_recommendations_total counter']
            for kind in ('personalized', 'popular'):
                lines.append(f'recsys_recommendations_total{{type="{kind}"}} {self.recommendations[kind]}')
        lines += [
            '# HELP recsys_uptime_seconds Время работы сервиса.',
            '# TYPE recsys_uptime_seconds gauge',
            f'recsys_uptime_seconds {time.time() - self.started:.0f}',
            '# HELP recsys_model_users Число пользователей с персональными рекомендациями.',
            '# TYPE recsys_model_users gauge',
            f'recsys_model_users {len(recommender.user_ids)}',
            '# HELP recsys_model_precision_at_3 Precision@3 модели на отложенном тестовом периоде.',
            '# TYPE recsys_model_precision_at_3 gauge',
            f'recsys_model_precision_at_3 {info["metrics"]["precision_at_3_test"]}',
            '# HELP recsys_model_info Версия модели.',
            '# TYPE recsys_model_info gauge',
            f'recsys_model_info{{version="{info["model_version"]}"}} 1',
        ]
        return '\n'.join(lines) + '\n'


class ApiError(Exception):
    """Ошибка в запросе клиента: возвращается как JSON с кодом 400."""

    def __init__(self, message, status=400):
        super().__init__(message)
        self.message = message
        self.status = status


def parse_user_id(value):
    """Проверяет идентификатор пользователя: целое неотрицательное число."""
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ApiError(f'user_id должен быть целым неотрицательным числом, получено: {value!r}')
    if isinstance(value, str):
        value = value.strip()
        if not value.isascii() or not value.isdigit() or len(value) > 19:
            raise ApiError(f'user_id должен быть целым неотрицательным числом, получено: {value!r}')
        value = int(value)
    if value < 0 or value > MAX_USER_ID:
        raise ApiError(f'user_id должен быть в диапазоне от 0 до {MAX_USER_ID}')
    return value


def parse_k(value):
    """Проверяет число рекомендаций: целое от 1 до MAX_K."""
    if value is None:
        return MAX_K
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ApiError(f'k должно быть целым числом от 1 до {MAX_K}')
    if isinstance(value, str):
        if not value.isascii() or not value.isdigit() or len(value) > 3:
            raise ApiError(f'k должно быть целым числом от 1 до {MAX_K}')
        value = int(value)
    if not 1 <= value <= MAX_K:
        raise ApiError(f'k должно быть целым числом от 1 до {MAX_K}')
    return value


def create_app(artifacts_dir=ARTIFACTS_DIR):
    app = Flask(__name__)
    app.json.ensure_ascii = False
    app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024        # тело запроса не больше 1 МБ
    recommender = Recommender(Path(artifacts_dir))
    metrics = Metrics()

    def make_recommendation(user_id, k):
        items, personalized = recommender.recommend(user_id, k)
        metrics.observe_recommendation(personalized)
        return {'user_id': user_id, 'items': items, 'personalized': personalized}

    @app.before_request
    def start_timer():
        g.started = time.perf_counter()

    @app.after_request
    def record_metrics(response):
        endpoint = request.url_rule.rule if request.url_rule else 'unknown'
        metrics.observe_request(endpoint, request.method, response.status_code,
                                time.perf_counter() - g.get('started', time.perf_counter()))
        return response

    @app.get('/')
    def index():
        return jsonify({
            'service': 'Рекомендации товаров для главной страницы',
            'model_version': recommender.info['model_version'],
            'endpoints': {
                'GET /recommend/<user_id>?k=3': 'рекомендации для одного пользователя',
                'POST /recommend': 'рекомендации для списка пользователей: {"user_ids": [1, 2], "k": 3}',
                'GET /health': 'проверка состояния сервиса',
                'GET /info': 'описание модели и её метрик',
                'GET /metrics': 'метрики сервиса в формате Prometheus',
            },
        })

    @app.get('/health')
    def health():
        return jsonify({'status': 'ok', 'model_version': recommender.info['model_version'],
                        'users': int(len(recommender.user_ids))})

    @app.get('/info')
    def info():
        return jsonify(recommender.info)

    @app.get('/metrics')
    def metrics_endpoint():
        return Response(metrics.render(recommender), mimetype='text/plain; version=0.0.4; charset=utf-8')

    @app.get('/recommend/<user_id>')
    def recommend_one(user_id):
        result = make_recommendation(parse_user_id(user_id), parse_k(request.args.get('k')))
        result['model_version'] = recommender.info['model_version']
        return jsonify(result)

    @app.post('/recommend')
    def recommend_many():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            raise ApiError('Тело запроса должно быть JSON-объектом вида {"user_ids": [1, 2, 3]}')
        user_ids = payload.get('user_ids')
        if not isinstance(user_ids, list) or not user_ids:
            raise ApiError('Поле user_ids должно быть непустым списком идентификаторов')
        if len(user_ids) > MAX_BATCH:
            raise ApiError(f'В одном запросе допускается не более {MAX_BATCH} пользователей')
        k = parse_k(payload.get('k'))
        results = [make_recommendation(parse_user_id(u), k) for u in user_ids]
        return jsonify({'model_version': recommender.info['model_version'], 'recommendations': results})

    @app.errorhandler(ApiError)
    def handle_api_error(error):
        return jsonify({'error': error.message}), error.status

    @app.errorhandler(HTTPException)
    def handle_http_error(error):
        messages = {404: 'Адрес не найден. Список адресов: GET /',
                    405: 'Метод не поддерживается для этого адреса',
                    413: 'Слишком большое тело запроса'}
        return jsonify({'error': messages.get(error.code, error.name)}), error.code

    @app.errorhandler(Exception)
    def handle_unexpected_error(error):
        app.logger.exception('Необработанная ошибка')
        return jsonify({'error': 'Внутренняя ошибка сервиса'}), 500

    return app


app = create_app()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8000)))
