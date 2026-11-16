"""Проверка сервиса: корректные и некорректные запросы.

Запуск из папки recsys:  python -m pytest tests  (или просто: python tests/test_app.py)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import MAX_BATCH, app  # noqa: E402

client = app.test_client()


def _first_user():
    import numpy as np
    data = np.load(Path(__file__).resolve().parents[1] / 'artifacts' / 'recommendations.npz')
    return int(data['user_ids'][0])


def test_health():
    r = client.get('/health')
    assert r.status_code == 200 and r.get_json()['status'] == 'ok'


def test_known_user_gets_three_items():
    r = client.get(f'/recommend/{_first_user()}')
    body = r.get_json()
    assert r.status_code == 200
    assert body['personalized'] is True
    assert len(body['items']) == 3 and len(set(body['items'])) == 3


def test_unknown_user_gets_popular_items():
    body = client.get('/recommend/999999999999').get_json()
    assert body['personalized'] is False and len(body['items']) == 3


def test_k_parameter():
    assert len(client.get(f'/recommend/{_first_user()}?k=1').get_json()['items']) == 1
    for bad in ('0', '4', '-1', 'abc', '1.5', ''):
        assert client.get(f'/recommend/1?k={bad}').status_code == 400, bad


def test_invalid_user_id():
    for bad in ('abc', '-5', '1.5', '1e3', '99999999999999999999999', '%20', "1;drop", '１２３'):
        r = client.get(f'/recommend/{bad}')
        assert r.status_code == 400, bad
        assert 'error' in r.get_json()


def test_batch():
    r = client.post('/recommend', json={'user_ids': [_first_user(), 999999999999, '7'], 'k': 2})
    body = r.get_json()
    assert r.status_code == 200 and len(body['recommendations']) == 3
    assert all(len(x['items']) == 2 for x in body['recommendations'])


def test_batch_invalid():
    bad_bodies = [None, [], {}, {'user_ids': []}, {'user_ids': 'abc'}, {'user_ids': [1, None]},
                  {'user_ids': [1.5]}, {'user_ids': [True]}, {'user_ids': [1], 'k': 10},
                  {'user_ids': list(range(MAX_BATCH + 1))}]
    for body in bad_bodies:
        assert client.post('/recommend', json=body).status_code == 400, body
    r = client.post('/recommend', data='{not json', content_type='application/json')
    assert r.status_code == 400
    r = client.post('/recommend', data='user_ids=1', content_type='text/plain')
    assert r.status_code == 400


def test_unknown_route_and_method():
    assert client.get('/nothing').status_code == 404
    assert client.delete('/recommend/1').status_code == 405
    assert client.get('/recommend').status_code == 405
    assert 'error' in client.get('/nothing').get_json()


def test_metrics():
    client.get('/recommend/1')
    text = client.get('/metrics').get_data(as_text=True)
    assert 'recsys_requests_total' in text
    assert 'recsys_recommendations_total{type="popular"}' in text
    assert 'recsys_model_precision_at_3' in text


if __name__ == '__main__':
    tests = [f for name, f in sorted(globals().items()) if name.startswith('test_')]
    for test in tests:
        test()
        print('ok', test.__name__)
    print(f'Все проверки пройдены: {len(tests)}')
