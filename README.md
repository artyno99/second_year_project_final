# Дипломный проект: рекомендательная система для интернет-магазина

Сервис по идентификатору пользователя возвращает три товара для блока рекомендаций на главной странице. Метрика качества — Precision@3; на отложенном тестовом периоде модель показывает 0,0199 против 0,0099 у рекомендации популярных товаров.

## Что в репозитории

| Файл или папка | Содержание |
|---|---|
| [`recsys/README.md`](recsys/README.md) | Документация: данные, факторы, валидация, эксперименты, запуск в Docker, API сервиса |
| [`second_year_final_project.ipynb`](second_year_final_project.ipynb) | Описание данных, построение валидации, генерация факторов |
| [`second_year_final_project_week3.ipynb`](second_year_final_project_week3.ipynb) | Эксперименты с моделями и выбор лучшей |
| [`recsys/`](recsys) | Код сервиса, скрипт обучения, Dockerfile, тесты, готовые рекомендации |
| [`recsys/recsys-service.tar`](recsys/recsys-service.tar) | Docker-образ сервиса, экспортированный командой `docker save` |
| [`presentation.pdf`](presentation.pdf) | Презентация с итогами проекта |

## Как запустить сервис

```bash
docker load -i recsys/recsys-service.tar
docker run -d --name recsys -p 8000:8000 recsys-service:1.0
curl http://localhost:8000/recommend/1000
```

Исходные данные (`events.csv`, `category_tree.csv`, `item_properties_part1.csv`, `item_properties_part2.csv`) в репозиторий не входят из-за размера.
