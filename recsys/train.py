"""Обучение модели и расчёт рекомендаций для сервиса.

Запуск (из папки recsys):

    python train.py --data-dir .. --out-dir artifacts

Скрипт выполняет два этапа:

1. Оценка качества. Модель обучается на срезах до 18 августа 2015 года и оценивается
   по Precision@3 на тестовом периоде (18 августа — 18 сентября).
2. Модель для сервиса. Модель с теми же гиперпараметрами переобучается на всех данных,
   для каждого пользователя с историей рассчитываются три рекомендации.

Результат — папка с артефактами:

    recommendations.npz — готовые рекомендации (их читает сервис);
    model_info.json     — описание модели, факторов и метрик;
    model.joblib        — обученная модель (для воспроизводимости, сервису не нужна).
"""
import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import (NOT_FEATURES, load_category_features, load_events, load_properties,
                      make_dataset, top_popular)

RANDOM_STATE = 42
SPLIT_DT = pd.Timestamp('2015-08-18')          # граница train / test
WINDOW = pd.Timedelta(days=31)                 # длина целевого периода
FIRST_CUTOFF = pd.Timestamp('2015-05-23')      # первый обучающий срез
STEP = pd.Timedelta(days=7)                    # шаг между срезами
NEG_FRAC = 0.2                                 # доля отрицательных примеров в обучении
LOGREG_C = 0.01                                # выбрано на валидации (см. ноутбук с экспериментами)
TOP_K = 3
N_POPULAR = 10


def log(message):
    print(f'[{time.strftime("%H:%M:%S")}] {message}', flush=True)


def make_model():
    return make_pipeline(SimpleImputer(strategy='median'), StandardScaler(),
                         LogisticRegression(C=LOGREG_C, max_iter=1000))


def train_cutoffs(last_allowed_end):
    """Границы обучающих срезов, целевой период которых заканчивается не позже last_allowed_end."""
    cutoffs, cutoff = [], FIRST_CUTOFF
    while cutoff + WINDOW <= last_allowed_end:
        cutoffs.append(cutoff)
        cutoff += STEP
    return cutoffs


def fit_model(events, props, cat_feats, cutoffs):
    slices = [make_dataset(events, props, cat_feats, c, window=WINDOW, neg_frac=NEG_FRAC,
                           random_state=RANDOM_STATE) for c in cutoffs]
    train = pd.concat(slices, ignore_index=True)
    feature_cols = [c for c in train.columns if c not in NOT_FEATURES]
    model = make_model().fit(train[feature_cols], train['target'])
    return model, feature_cols, len(train), int(train['target'].sum())


def rank_candidates(model, feature_cols, data):
    """Кандидаты с оценкой модели, упорядоченные по убыванию оценки внутри пользователя."""
    scored = data[['visitorid', 'itemid']].assign(score=model.predict_proba(data[feature_cols])[:, 1])
    scored = scored.sort_values(['visitorid', 'score'], ascending=[True, False])
    scored['rank'] = scored.groupby('visitorid').cumcount()
    return scored[scored['rank'] < TOP_K]


def fill_with_default(items, default, k=TOP_K):
    """Дополняет список рекомендаций популярными товарами до k штук без повторов."""
    return list(dict.fromkeys(list(items) + list(default)))[:k]


def precision_at_k(recs, actual, default, k=TOP_K):
    return float(np.mean([len(set(recs.get(u, default)[:k]) & bought) / k for u, bought in actual.items()]))


def evaluate(events, props, cat_feats):
    """Precision@3 на тестовом периоде для модели, обученной на данных до SPLIT_DT."""
    cutoffs = train_cutoffs(SPLIT_DT)
    model, feature_cols, n_rows, n_pos = fit_model(events, props, cat_feats, cutoffs)

    test = events[(events['event_dt'] >= SPLIT_DT) & (events['event'] == 'transaction')]
    actual = test.groupby('visitorid')['itemid'].agg(set).to_dict()
    default = top_popular(events, SPLIT_DT, k=TOP_K)
    data = make_dataset(events, props, cat_feats, SPLIT_DT, users=set(actual))
    top = rank_candidates(model, feature_cols, data)
    recs = {user: fill_with_default(items.tolist(), default) for user, items in top.groupby('visitorid')['itemid']}
    return {
        'precision_at_3_test': round(precision_at_k(recs, actual, default), 5),
        'precision_at_3_test_popular_baseline': round(precision_at_k({}, actual, default), 5),
        'test_period': [str(SPLIT_DT.date()), str(events['event_dt'].max().date())],
        'test_buyers': len(actual),
        'test_buyers_with_history': len(recs),
        'train_slices': len(cutoffs),
        'train_rows': n_rows,
        'train_positive': n_pos,
    }


def build_recommendations(events, props, cat_feats):
    """Обучает модель на всех данных и считает рекомендации для всех пользователей с историей."""
    data_end = events['event_dt'].max() + pd.Timedelta(seconds=1)
    cutoffs = train_cutoffs(data_end)
    model, feature_cols, n_rows, n_pos = fit_model(events, props, cat_feats, cutoffs)
    log(f'Модель обучена: срезов {len(cutoffs)}, строк {n_rows}, положительных {n_pos}')

    popular = top_popular(events, data_end, k=N_POPULAR)
    data = make_dataset(events, props, cat_feats, data_end)
    top = rank_candidates(model, feature_cols, data)

    user_ids = np.sort(top['visitorid'].unique())
    items = np.full((len(user_ids), TOP_K), -1, dtype=np.int32)
    rows = np.searchsorted(user_ids, top['visitorid'].to_numpy())
    items[rows, top['rank'].to_numpy()] = top['itemid'].to_numpy()

    # свободные места заполняем популярными товарами (без повторов)
    for row in np.where((items < 0).any(axis=1))[0]:
        own = [i for i in items[row] if i >= 0]
        items[row] = fill_with_default(own, popular)

    info = {
        'data_period': [str(events['event_dt'].min().date()), str(events['event_dt'].max().date())],
        'production_train_slices': len(cutoffs),
        'production_train_rows': n_rows,
        'production_train_positive': n_pos,
        'users_with_personal_recommendations': int(len(user_ids)),
        'popular_items': [int(i) for i in popular],
        'features': feature_cols,
    }
    return model, user_ids.astype(np.int64), items, np.array(popular, dtype=np.int32), info


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data-dir', default='..', help='папка с events.csv, category_tree.csv, item_properties_part*.csv')
    parser.add_argument('--out-dir', default='artifacts', help='папка для артефактов')
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log('Загрузка данных')
    events = load_events(args.data_dir)
    props = load_properties(args.data_dir)
    cat_feats = load_category_features(args.data_dir)
    log(f'Событий: {len(events)}, записей о свойствах: {len(props)}')

    log('Этап 1: оценка качества на тестовом периоде')
    metrics = evaluate(events, props, cat_feats)
    log(f'Precision@3 на тесте: {metrics["precision_at_3_test"]} '
        f'(популярные товары: {metrics["precision_at_3_test_popular_baseline"]})')

    log('Этап 2: обучение на всех данных и расчёт рекомендаций')
    model, user_ids, items, popular, info = build_recommendations(events, props, cat_feats)

    np.savez_compressed(out_dir / 'recommendations.npz', user_ids=user_ids, items=items, popular=popular)
    joblib.dump(model, out_dir / 'model.joblib')
    model_info = {
        'model_version': datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S'),
        'model': 'LogisticRegression (ранжирование товаров из истории пользователя) + популярные товары',
        'hyperparameters': {'C': LOGREG_C, 'imputer': 'median', 'scaler': 'StandardScaler',
                            'target_window_days': WINDOW.days, 'negative_sampling': NEG_FRAC},
        'metrics': metrics,
        **info,
    }
    with open(out_dir / 'model_info.json', 'w', encoding='utf-8') as f:
        json.dump(model_info, f, ensure_ascii=False, indent=2)
    log(f'Готово. Пользователей с персональными рекомендациями: {len(user_ids)}. Артефакты: {out_dir.resolve()}')


if __name__ == '__main__':
    main()
