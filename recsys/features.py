"""Загрузка данных и генерация факторов для рекомендательной системы.

Все функции генерации факторов получают момент времени ``cutoff`` и используют
только события и свойства товаров, записанные до этого момента.
"""
from pathlib import Path

import numpy as np
import pandas as pd

EVENT_TYPES = ['view', 'addtocart', 'transaction']
KEEP_PROPS = ['categoryid', 'available', '790']
PROP_DTYPES = {'timestamp': 'int64', 'itemid': 'int32', 'property': 'str', 'value': 'str'}
NOT_FEATURES = ['visitorid', 'itemid', 'target', 'categoryid', 'cat_parent', 'cat_root']


# ---------------------------------------------------------------------------
# Загрузка данных
# ---------------------------------------------------------------------------

def load_events(data_dir):
    """Лог событий без полных дубликатов, с колонкой event_dt."""
    events = pd.read_csv(Path(data_dir) / 'events.csv')
    events = events.drop_duplicates().sort_values('timestamp').reset_index(drop=True)
    events['event_dt'] = pd.to_datetime(events['timestamp'], unit='ms')
    return events


def load_properties(data_dir, chunksize=2_000_000):
    """Свойства товаров: читаются частями, остаются только свойства из KEEP_PROPS."""
    kept = []
    for name in ('item_properties_part1.csv', 'item_properties_part2.csv'):
        for chunk in pd.read_csv(Path(data_dir) / name, chunksize=chunksize, dtype=PROP_DTYPES):
            kept.append(chunk[chunk['property'].isin(KEEP_PROPS)])
    props = pd.concat(kept, ignore_index=True)
    props['prop_dt'] = pd.to_datetime(props['timestamp'], unit='ms')
    return props


def load_category_features(data_dir):
    """Дерево категорий: родитель, корень и глубина каждой категории."""
    cat_tree = pd.read_csv(Path(data_dir) / 'category_tree.csv')
    parent_of = cat_tree.dropna(subset=['parentid']).set_index('categoryid')['parentid'].astype(int).to_dict()

    def path_to_root(cat):
        depth, seen = 0, set()
        while cat in parent_of and cat not in seen:
            seen.add(cat)
            cat = parent_of[cat]
            depth += 1
        return cat, depth

    cat_feats = cat_tree.copy()
    cat_feats[['cat_root', 'cat_depth']] = [path_to_root(c) for c in cat_feats['categoryid']]
    cat_feats = cat_feats.rename(columns={'parentid': 'cat_parent'})
    cat_feats['cat_parent'] = cat_feats['cat_parent'].fillna(-1).astype(int)
    return cat_feats


# ---------------------------------------------------------------------------
# Генерация факторов
# ---------------------------------------------------------------------------

def count_events(df, keys, prefix):
    """Число просмотров, корзин и покупок в разрезе keys."""
    out = (df.groupby(keys + ['event']).size().unstack(fill_value=0)
             .reindex(columns=EVENT_TYPES, fill_value=0))
    out.columns = [f'{prefix}_views', f'{prefix}_carts', f'{prefix}_trans']
    return out.astype('int32')


def days_between(later, earlier):
    return ((later - earlier).dt.total_seconds() / 86400).astype('float32')


def property_asof(props, name, cutoff):
    """Последнее значение свойства name для каждого товара, записанное до cutoff."""
    p = props[(props['property'] == name) & (props['prop_dt'] < cutoff)].sort_values('timestamp')
    return p.groupby('itemid')['value'].last()


def build_item_features(ev, cutoff, props, cat_feats):
    """Факторы товара по истории ev (только события до cutoff)."""
    f = count_events(ev, ['itemid'], 'item')
    f['item_users'] = ev.groupby('itemid')['visitorid'].nunique().astype('int32')
    f['item_cart_rate'] = (f['item_carts'] / (f['item_views'] + 1)).astype('float32')
    f['item_buy_rate'] = (f['item_trans'] / (f['item_views'] + 1)).astype('float32')
    for days in (7, 30):
        recent = ev[ev['event_dt'] >= cutoff - pd.Timedelta(days=days)]
        f = f.join(count_events(recent, ['itemid'], f'item_{days}d'), how='left')
    f = f.fillna(0)
    g = ev.groupby('itemid')['event_dt']
    f['item_days_since_last'] = days_between(cutoff, g.max())
    f['item_days_since_first'] = days_between(cutoff, g.min())
    f['categoryid'] = pd.to_numeric(property_asof(props, 'categoryid', cutoff), errors='coerce')
    f['item_available'] = pd.to_numeric(property_asof(props, 'available', cutoff), errors='coerce')
    p790 = property_asof(props, '790', cutoff)
    f['item_prop_790_log'] = np.log1p(pd.to_numeric(p790.str[1:], errors='coerce')).astype('float32')
    f = f.reset_index().merge(cat_feats, on='categoryid', how='left')
    cat_pop = f.groupby('categoryid')[['item_views', 'item_trans']].sum()
    cat_pop.columns = ['cat_views', 'cat_trans']
    f = f.merge(cat_pop.reset_index(), on='categoryid', how='left')
    f['cat_items'] = f.groupby('categoryid')['itemid'].transform('count')
    return f


def build_user_features(ev, cutoff):
    """Факторы пользователя по истории ev (только события до cutoff)."""
    f = count_events(ev, ['visitorid'], 'user')
    f['user_items'] = ev.groupby('visitorid')['itemid'].nunique().astype('int32')
    days = ev.assign(day=ev['event_dt'].dt.normalize()).drop_duplicates(['visitorid', 'day'])
    f['user_active_days'] = days.groupby('visitorid').size().astype('int32')
    f['user_cart_rate'] = (f['user_carts'] / (f['user_views'] + 1)).astype('float32')
    f['user_buy_rate'] = (f['user_trans'] / (f['user_views'] + 1)).astype('float32')
    g = ev.groupby('visitorid')['event_dt']
    f['user_days_since_last'] = days_between(cutoff, g.max())
    f['user_days_since_first'] = days_between(cutoff, g.min())
    return f.reset_index()


def build_user_item_features(ev, cutoff, item_cat):
    """Факторы пары «пользователь–товар» и пары «пользователь–категория»."""
    ui = count_events(ev, ['visitorid', 'itemid'], 'ui')
    g = ev.groupby(['visitorid', 'itemid'])['event_dt']
    ui['ui_days_since_last'] = days_between(cutoff, g.max())
    ui['ui_days_since_first'] = days_between(cutoff, g.min())
    days = ev.assign(day=ev['event_dt'].dt.normalize()).drop_duplicates(['visitorid', 'itemid', 'day'])
    ui['ui_active_days'] = days.groupby(['visitorid', 'itemid']).size().astype('int32')
    ui['ui_cart_no_buy'] = ((ui['ui_carts'] > 0) & (ui['ui_trans'] == 0)).astype('int8')
    ui = ui.reset_index()
    # место товара в истории пользователя: 1 — товар, с которым он взаимодействовал последним
    ui['ui_recency_rank'] = ui.groupby('visitorid')['ui_days_since_last'].rank(method='first').astype('float32')

    ev_cat = ev[['visitorid', 'itemid', 'event']].merge(item_cat, on='itemid', how='inner')
    uc = count_events(ev_cat, ['visitorid', 'categoryid'], 'uc').reset_index()
    return ui, uc


def build_features(events, cutoff, props, cat_feats):
    """Считает все группы факторов по событиям строго до cutoff."""
    ev = events[events['event_dt'] < cutoff]
    item_f = build_item_features(ev, cutoff, props, cat_feats)
    user_f = build_user_features(ev, cutoff)
    item_cat = item_f[['itemid', 'categoryid']].dropna()
    ui_f, uc_f = build_user_item_features(ev, cutoff, item_cat)
    return {'item': item_f, 'user': user_f, 'user_item': ui_f, 'user_category': uc_f}


def assemble_dataset(pairs, feats):
    """Собирает таблицу факторов для заданных пар «пользователь–товар»."""
    df = (pairs.merge(feats['user_item'], on=['visitorid', 'itemid'], how='left')
               .merge(feats['user'], on='visitorid', how='left')
               .merge(feats['item'], on='itemid', how='left')
               .merge(feats['user_category'], on=['visitorid', 'categoryid'], how='left'))
    count_cols = [c for c in df.columns if c.startswith(('ui_', 'uc_')) and 'days' not in c]
    df[count_cols] = df[count_cols].fillna(0)
    df['ui_views_share'] = (df['ui_views'] / (df['user_views'] + 1)).astype('float32')
    return df


def make_dataset(events, props, cat_feats, cutoff, window=None, users=None, neg_frac=None, random_state=42):
    """Таблица факторов для кандидатов (пар из истории до cutoff).

    window   — длина целевого периода; если задана, добавляется колонка target
               (купил ли пользователь товар в период [cutoff, cutoff + window));
    users    — если задано, кандидаты берутся только для этих пользователей;
    neg_frac — доля отрицательных примеров, которую нужно оставить (None — все).
    """
    hist = events[events['event_dt'] < cutoff]
    candidates = hist[['visitorid', 'itemid']].drop_duplicates()
    if users is not None:
        candidates = candidates[candidates['visitorid'].isin(users)]
    if window is not None:
        future = events[(events['event_dt'] >= cutoff) & (events['event_dt'] < cutoff + window)
                        & (events['event'] == 'transaction')]
        target = future[['visitorid', 'itemid']].drop_duplicates().assign(target=1)
        candidates = candidates.merge(target, on=['visitorid', 'itemid'], how='left')
        candidates['target'] = candidates['target'].fillna(0).astype('int8')
        if neg_frac is not None:
            negatives = candidates[candidates['target'] == 0].sample(frac=neg_frac, random_state=random_state)
            candidates = pd.concat([candidates[candidates['target'] == 1], negatives], ignore_index=True)
    feats = build_features(events, cutoff, props, cat_feats)
    return assemble_dataset(candidates, feats)


def top_popular(events, cutoff, k=3, days=30):
    """Самые продаваемые товары за последние days дней перед cutoff."""
    recent = events[(events['event'] == 'transaction') & (events['event_dt'] < cutoff)
                    & (events['event_dt'] >= cutoff - pd.Timedelta(days=days))]
    return recent['itemid'].value_counts().head(k).index.tolist()
