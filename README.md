# 💳 Credit Risk Classification Project

Проект по анализу и прогнозированию кредитного риска заемщиков с использованием методов машинного обучения.
Модель оценивает вероятность дефолта и преобразует её в интерпретируемый кредитный рейтинг.

---

## 💡 Пример реализации

<table>
  <tr>
    <td><img src="https://github.com/golovanov-andrei/Credit-risk/main/screens/streamlit.png" width="100%"></td>
    <td><img src="https://github.com/golovanov-andrei/Credit-risk/main/screens/result.png" width="100%"></td>
  </tr>
  <tr>
    <td><img src="https://github.com/golovanov-andrei/Credit-risk/main/screens/collab.png" width="100%"></td>
    
  </tr>

---

## 🚀 Ключевые возможности

* **End-to-End Pipeline**
  Полный цикл: очистка данных, feature engineering, масштабирование и инференс.

* **ML-модель**
  Логистическая регрессия с учётом дисбаланса классов (`class_weight='balanced'`).

* **Risk Scoring System**
  Преобразование вероятности дефолта в категории риска (A / B / C).

* **Web-приложение**
  Интерфейс на Streamlit для интерактивной оценки заемщиков.

---

## 🛠 Технологический стек

* **Python 3.x**
* **pandas, numpy** — обработка данных
* **scikit-learn** — обучение модели
* **matplotlib, seaborn** — визуализация
* **joblib** — сохранение модели
* **streamlit** — веб-интерфейс
* **shap** — интерпретация предсказаний

---

## 📈 Этапы реализации

### 1. Feature Engineering

Созданы новые признаки:

* `saving_score`, `checking_score` — кодирование состояния счетов
* `job_risk` — индекс стабильности работы
* `has_own_house`, `is_rent` — тип жилья
* One-Hot Encoding категориальных признаков

---

### 2. Обучение модели

Использован `Pipeline`:

* `StandardScaler`
* `LogisticRegression`

Особенности:

* учтён дисбаланс классов
* кастомный порог классификации: **0.3**
* оптимизация под **F1-score**

Метрики:

* **ROC-AUC**
* **F1-score**

---

### 3. Система риск-классов

* **A (p < 0.2)** — низкий риск → одобрено
* **B (0.2 ≤ p < 0.5)** — средний риск → требуется внимание
* **C (p ≥ 0.5)** — высокий риск → отказ

---

## 🖥 Использование

### Установка зависимостей

```bash
pip install -r requirements.txt
```

### Обучение модели

Откройте и запустите ноутбук:

```bash
Credit risk.ipynb
```

### Запуск приложения

```bash
streamlit run app.py
```

---

## 👤 Автор

**Андрей Голованов**
GitHub: https://github.com/golovanov-andrei
