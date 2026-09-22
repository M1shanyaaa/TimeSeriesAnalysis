"""
Лабораторна робота №2.
Статистичне навчання за Big Data Time Series.

Вхід:
    results/parsed_data.csv з лабораторної №1.
    Обов'язкові колонки: date, cpi.

Реалізовано:
    1. Поліноміальна регресія МНК.
    2. Вибір моделі за часовою крос-валідацією.
    3. Прогноз на ceil(0.5 * n) місяців.
    4. Hampel-фільтр як базовий метод.
    5. Експериментальний адаптивний детектор:
       - два масштаби локального аналізу;
       - узгодженість напрямку відхилень;
       - захист послідовних змін;
       - навчання параметрів на контрольованих ін'єкціях.
    6. Незалежний часовий holdout.
    7. Програмні тести, таблиці, графіки та підсумок.

Важливо:
    Потенційна аномалія ІСЦ не обов'язково є помилкою вимірювання.
    Очищені значення використовуються тільки як експериментальний
    варіант навчальних даних. Оригінальний ряд не змінюється.

Встановлення:
    pip install numpy pandas matplotlib

Запуск:
    python lab2.py
    python lab2.py --input results/parsed_data.csv
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SEED = 42
METHODS = ("raw", "hampel", "adaptive")


# ============================================================
# Допоміжні функції
# ============================================================

def save_json(path, data):
    Path(path).write_text(
        json.dumps(
            data,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )


def save_table(path, table):
    table.to_csv(path, index=False, encoding="utf-8-sig")


def load_series(filename):
    """Читання реальних даних, отриманих у лабораторній №1."""

    df = pd.read_csv(filename)

    required = {"date", "cpi"}
    if not required.issubset(df.columns):
        raise ValueError("CSV повинен містити колонки date та cpi.")

    df = df[["date", "cpi"]].copy()
    df["date"] = pd.to_datetime(df["date"], errors="raise")
    df["cpi"] = pd.to_numeric(df["cpi"], errors="raise")
    df = df.sort_values("date").reset_index(drop=True)

    if df.empty:
        raise ValueError("Вхідний файл порожній.")

    if df["date"].duplicated().any():
        raise ValueError("Знайдено дублікати дат.")

    if not np.isfinite(df["cpi"].to_numpy()).all():
        raise ValueError("Ряд містить NaN або нескінченні значення.")

    if (df["cpi"] <= 0).any():
        raise ValueError("ІСЦ повинен бути додатним.")

    if len(df) < 96:
        raise ValueError(
            "Для цього експерименту потрібно щонайменше "
            "96 щомісячних спостережень."
        )

    expected = pd.date_range(
        df["date"].iloc[0],
        periods=len(df),
        freq="MS",
    )

    if not np.array_equal(
        expected.to_numpy(),
        df["date"].to_numpy(),
    ):
        raise ValueError(
            "Потрібен неперервний щомісячний ряд із датами "
            "першого числа місяця."
        )

    return df


def regression_metrics(actual, predicted):
    """Метрики прогнозу. R² не є ймовірністю точного прогнозу."""

    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    if actual.shape != predicted.shape:
        raise ValueError("Розміри actual і predicted повинні збігатися.")

    error = actual - predicted
    mse = float(np.mean(error ** 2))
    denominator = float(np.sum((actual - actual.mean()) ** 2))

    return {
        "MAE": float(np.mean(np.abs(error))),
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "R2": (
            float(1 - np.sum(error ** 2) / denominator)
            if denominator > 0 else None
        ),
    }


def binary_metrics(actual, predicted):
    """Метрики лише для відомих штучно внесених аномалій."""

    actual = np.asarray(actual, dtype=bool)
    predicted = np.asarray(predicted, dtype=bool)

    tp = int(np.sum(actual & predicted))
    fp = int(np.sum(~actual & predicted))
    fn = int(np.sum(actual & ~predicted))

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# ============================================================
# Робастні локальні характеристики
# ============================================================

def robust_scale(values):
    """Робастна оцінка масштабу через MAD."""

    values = np.asarray(values, dtype=float)
    center = np.median(values)
    mad = np.median(np.abs(values - center))
    return float(1.4826 * mad)


def local_profile(y, window):
    """
    Локальна медіана та MAD.

    Центр вікна не включаємо до опорної вибірки:
    великий викид не повинен сам збільшувати свій поріг.

    Центровані вікна використовуються лише всередині
    доступного навчального фрагмента.
    """

    y = np.asarray(y, dtype=float)

    if window < 3 or window % 2 == 0:
        raise ValueError("Розмір вікна повинен бути непарним і >= 3.")

    n = len(y)
    half = window // 2

    medians = np.empty(n)
    scales = np.empty(n)

    if n > 1:
        global_noise = robust_scale(np.diff(y)) / np.sqrt(2)
    else:
        global_noise = 0.0

    # Нижня межа захищає від ділення на нуль та
    # надмірної чутливості на майже сталих ділянках.
    numerical_floor = 1e-8 * max(1.0, float(np.max(np.abs(y))))
    scale_floor = max(0.25 * global_noise, numerical_floor)

    for i in range(n):
        left = max(0, i - half)
        right = min(n, i + half + 1)

        neighbors = np.concatenate([
            y[left:i],
            y[i + 1:right],
        ])

        if len(neighbors) == 0:
            medians[i] = y[i]
            scales[i] = scale_floor
            continue

        medians[i] = np.median(neighbors)
        scales[i] = max(robust_scale(neighbors), scale_floor)

    return medians, scales


def protect_persistent_changes(mask, residual, minimum_run=3):
    """
    Не очищати серії з >= minimum_run підозрілих точок
    однакового напрямку.

    Гіпотеза:
        тривала односпрямована зміна може бути економічною
        подією або зміною режиму, а не одиночним збоєм.

    Обмеження:
        через цей захист алгоритм може пропустити
        пакет послідовних помилкових вимірів.
    """

    mask = np.asarray(mask, dtype=bool)
    result = mask.copy()
    n = len(mask)
    i = 0

    while i < n:
        if not mask[i]:
            i += 1
            continue

        j = i + 1
        direction = np.sign(residual[i])

        while (
            j < n
            and mask[j]
            and np.sign(residual[j]) == direction
        ):
            j += 1

        if j - i >= minimum_run:
            result[i:j] = False

        i = j

    return result


# ============================================================
# Методи очищення
# ============================================================

def clean_series(y, method, window=13, q=3.0):
    """
    Повертає:
        cleaned — копію ряду після експериментального очищення;
        mask    — ознаки потенційних аномалій;
        score   — стандартизовані оцінки аномальності.

    Оригінальний масив не змінюється.
    """

    y = np.asarray(y, dtype=float)
    cleaned = y.copy()

    if method == "raw":
        return (
            cleaned,
            np.zeros(len(y), dtype=bool),
            np.zeros(len(y), dtype=float),
        )

    median_short, scale_short = local_profile(y, window)
    residual_short = y - median_short
    score_short = np.abs(residual_short) / scale_short

    if method == "hampel":
        score = score_short
        mask = score > q
        replacement = median_short

    elif method == "adaptive":
        long_window = 2 * window - 1

        median_long, scale_long = local_profile(y, long_window)
        residual_long = y - median_long
        score_long = np.abs(residual_long) / scale_long

        # Відхилення має бути значним на обох масштабах.
        score = np.minimum(score_short, score_long)

        # На обох масштабах відхилення повинно мати один знак.
        agreement = residual_short * residual_long > 0

        candidates = (score > q) & agreement

        mask = protect_persistent_changes(
            candidates,
            residual_short,
            minimum_run=3,
        )

        # Короткий масштаб точніше відтворює локальну динаміку.
        replacement = median_short

    else:
        raise ValueError(f"Невідомий метод: {method}")

    cleaned[mask] = replacement[mask]
    return cleaned, mask, score


# ============================================================
# R&D: навчання параметрів через штучні ін'єкції
# ============================================================

def inject_anomalies(y, rng, rate=0.05):
    """
    Внести відомі одиночні аномалії в копію ряду.

    Оригінал є ціллю відновлення, але не оголошується
    фізично безпомилковим рядом.
    """

    y = np.asarray(y, dtype=float)
    contaminated = y.copy()

    # Не використовуємо крайні точки для калібрувальних ін'єкцій.
    available = np.arange(2, len(y) - 2)

    if len(available) == 0:
        raise ValueError("Ряд надто короткий для внесення аномалій.")

    count = max(1, int(round(rate * len(y))))
    count = min(count, len(available))

    indices = np.sort(
        rng.choice(available, size=count, replace=False)
    )

    noise_scale = robust_scale(np.diff(y)) / np.sqrt(2)
    noise_scale = max(
        noise_scale,
        0.1 * robust_scale(y),
        0.05,
    )

    signs = rng.choice([-1.0, 1.0], size=count)
    amplitudes = rng.uniform(5.0, 9.0, size=count) * noise_scale

    contaminated[indices] += signs * amplitudes

    labels = np.zeros(len(y), dtype=bool)
    labels[indices] = True

    return contaminated, labels


def calibrate_detector(y, repeats=12):
    """
    Навчити window та q тільки на початковій частині
    development-вибірки.

    Критерій:
        мінімальний середній RMSE відновлення початкового
        ряду після контрольованого внесення аномалій.

    Калібрувальні реалізації однакові для всіх кандидатів.
    """

    rng = np.random.default_rng(SEED + 10)

    cases = [
        inject_anomalies(y, rng)
        for _ in range(repeats)
    ]

    rows = []

    for window in (7, 13, 19):
        for q in (2.5, 3.0, 3.5, 4.5):
            errors = []
            f1_values = []

            for contaminated, labels in cases:
                cleaned, mask, _ = clean_series(
                    contaminated,
                    method="adaptive",
                    window=window,
                    q=q,
                )

                errors.append(
                    regression_metrics(y, cleaned)["RMSE"]
                )
                f1_values.append(binary_metrics(labels, mask)["f1"])

            rows.append({
                "window": window,
                "q": q,
                "reconstruction_rmse": float(np.mean(errors)),
                "injected_f1": float(np.mean(f1_values)),
            })

    table = pd.DataFrame(rows).sort_values(
        ["reconstruction_rmse", "window", "q"]
    ).reset_index(drop=True)

    best = table.iloc[0]

    parameters = {
        "window": int(best["window"]),
        "q": float(best["q"]),
    }

    return parameters, table


def method_parameters(method, adaptive_parameters):
    if method == "adaptive":
        return adaptive_parameters

    return {"window": 13, "q": 3.0}


def injection_benchmark(y, adaptive_parameters, repeats=20):
    """
    Нові ін'єкції на пізнішому development-фрагменті.

    Ця таблиця не використовується для вибору параметрів.
    Це перевірка стійкості до додаткових штучних збурень,
    а не розмітка справжніх економічних аномалій.
    """

    rng = np.random.default_rng(SEED + 20)
    cases = [inject_anomalies(y, rng) for _ in range(repeats)]
    rows = []

    for method in METHODS:
        errors = []
        precision = []
        recall = []
        f1 = []

        parameters = method_parameters(method, adaptive_parameters)

        for contaminated, labels in cases:
            cleaned, mask, _ = clean_series(
                contaminated,
                method,
                **parameters,
            )

            errors.append(
                regression_metrics(y, cleaned)["RMSE"]
            )

            detection = binary_metrics(labels, mask)
            precision.append(detection["precision"])
            recall.append(detection["recall"])
            f1.append(detection["f1"])

        rows.append({
            "method": method,
            "reconstruction_rmse": float(np.mean(errors)),
            "injected_precision": float(np.mean(precision)),
            "injected_recall": float(np.mean(recall)),
            "injected_f1": float(np.mean(f1)),
        })

    return pd.DataFrame(rows)


# ============================================================
# Поліноміальна регресія МНК
# ============================================================

def design_matrix(t, degree, seasonal, center, scale):
    """
    Поліном від масштабованого часу.

    За seasonal=True додаються два гармонічні регресори.
    Модель залишається лінійною за коефіцієнтами
    та навчається методом найменших квадратів.
    """

    t = np.asarray(t, dtype=float)
    z = (t - center) / scale

    columns = [z ** power for power in range(degree + 1)]

    if seasonal:
        columns.extend([
            np.sin(2 * np.pi * t / 12),
            np.cos(2 * np.pi * t / 12),
        ])

    return np.column_stack(columns)


def fit_lsm(t, y, degree, seasonal):
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)

    center = float(t.mean())
    scale = max(float(np.ptp(t)) / 2, 1.0)

    X = design_matrix(t, degree, seasonal, center, scale)

    # Стійкіше, ніж обчислення inv(X.T @ X).
    beta, _, rank, _ = np.linalg.lstsq(X, y, rcond=None)

    if rank < X.shape[1]:
        raise ValueError("Матриця регресії має неповний ранг.")

    return {
        "degree": int(degree),
        "seasonal": bool(seasonal),
        "center": center,
        "scale": scale,
        "coefficients": beta.tolist(),
    }


def predict_lsm(model, t):
    X = design_matrix(
        t,
        model["degree"],
        model["seasonal"],
        model["center"],
        model["scale"],
    )

    prediction = X @ np.asarray(model["coefficients"])

    if not np.isfinite(prediction).all():
        raise ValueError("Модель сформувала некоректний прогноз.")

    return prediction


# ============================================================
# Часова крос-валідація
# ============================================================

def temporal_folds(n):
    """
    Навчальне вікно розширюється.
    Майбутні точки не потрапляють у навчальну частину.
    """

    validation_size = max(6, int(n * 0.15))

    for fraction in (0.50, 0.65, 0.80):
        stop = int(n * fraction)
        finish = min(n, stop + validation_size)

        if finish > stop:
            yield stop, finish


def choose_models(y, adaptive_parameters):
    """
    Вибір:
        метод очищення;
        степінь полінома 0..3;
        наявність річної гармоніки.

    Validation оцінюється на ОРИГІНАЛЬНИХ значеннях:
    очищення не може штучно спростити перевірку прогнозу.
    """

    t = np.arange(len(y), dtype=float)
    rows = []
    fold_rows = []

    for method in METHODS:
        parameters = method_parameters(method, adaptive_parameters)

        for degree in range(4):
            for seasonal in (False, True):
                squared_errors = []

                for fold_number, (stop, finish) in enumerate(
                    temporal_folds(len(y)), start=1
                ):
                    train_clean, mask, _ = clean_series(
                        y[:stop],
                        method,
                        **parameters,
                    )

                    model = fit_lsm(
                        t[:stop],
                        train_clean,
                        degree,
                        seasonal,
                    )

                    predicted = predict_lsm(model, t[stop:finish])
                    actual = y[stop:finish]

                    squared_errors.extend(
                        ((actual - predicted) ** 2).tolist()
                    )

                    fold_rows.append({
                        "method": method,
                        "degree": degree,
                        "seasonal": seasonal,
                        "fold": fold_number,
                        "train_n": stop,
                        "validation_n": finish - stop,
                        "flagged_train": int(mask.sum()),
                        "RMSE": regression_metrics(
                            actual, predicted
                        )["RMSE"],
                    })

                rows.append({
                    "method": method,
                    "degree": degree,
                    "seasonal": seasonal,
                    "cv_rmse": float(
                        np.sqrt(np.mean(squared_errors))
                    ),
                })

    scores = pd.DataFrame(rows).sort_values(
        ["cv_rmse", "degree", "seasonal", "method"]
    ).reset_index(drop=True)

    return scores, pd.DataFrame(fold_rows)


def config_from_row(row):
    return {
        "method": str(row["method"]),
        "degree": int(row["degree"]),
        "seasonal": bool(row["seasonal"]),
    }


def fit_pipeline(y, config, adaptive_parameters):
    parameters = method_parameters(
        config["method"],
        adaptive_parameters,
    )

    cleaned, mask, score = clean_series(
        y,
        config["method"],
        **parameters,
    )

    model = fit_lsm(
        np.arange(len(y), dtype=float),
        cleaned,
        config["degree"],
        config["seasonal"],
    )

    return model, cleaned, mask, score


# ============================================================
# Перевірка на незалежному часовому holdout
# ============================================================

def evaluate_holdout(
    y,
    split,
    scores,
    adaptive_parameters,
    selected_method,
):
    """
    Для кожного методу беремо найкращу конфігурацію за CV.

    Test використовується лише для звіту:
    за його результатами модель НЕ перевибирається.
    """

    train = y[:split]
    actual = y[split:]
    future_t = np.arange(split, len(y), dtype=float)

    rows = []
    predictions = {}

    for method in METHODS:
        best_row = scores.loc[scores["method"] == method].iloc[0]
        config = config_from_row(best_row)

        model, _, mask, _ = fit_pipeline(
            train, config, adaptive_parameters
        )
        predicted = predict_lsm(model, future_t)
        predictions[method] = predicted

        rows.append({
            **config,
            "cv_rmse": float(best_row["cv_rmse"]),
            "selected_by_cv": method == selected_method,
            "flagged_train": int(mask.sum()),
            **regression_metrics(actual, predicted),
        })

    # Дві прості моделі для контролю якості.
    naive = np.full(len(actual), train[-1])

    seasonal_naive = np.array([
        train[split - 12 + (i % 12)]
        for i in range(len(actual))
    ])

    for name, predicted in (
        ("naive_last", naive),
        ("seasonal_naive", seasonal_naive),
    ):
        predictions[name] = predicted
        rows.append({
            "method": name,
            "degree": None,
            "seasonal": None,
            "cv_rmse": None,
            "selected_by_cv": False,
            "flagged_train": 0,
            **regression_metrics(actual, predicted),
        })

    return pd.DataFrame(rows), predictions


# ============================================================
# Візуалізація
# ============================================================

def make_plots(
    df,
    cleaned_variants,
    masks,
    scores,
    calibration,
    benchmark,
    split,
    holdout_predictions,
    fitted,
    future_dates,
    forecast,
    selected,
    output,
):
    dates = df["date"]
    y = df["cpi"].to_numpy()

    plt.rcParams.update({
        "font.size": 10,
        "axes.grid": True,
        "grid.alpha": 0.25,
    })

    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)

    for ax, method in zip(axes, ("hampel", "adaptive")):
        mask = masks[method]

        ax.plot(dates, y, label="Оригінальний ІСЦ", alpha=0.7)
        ax.plot(
            dates,
            cleaned_variants[method],
            label=f"Після {method}",
            linewidth=1.7,
        )
        ax.scatter(
            dates[mask],
            y[mask],
            color="red",
            marker="x",
            s=55,
            label="Потенційні аномалії",
            zorder=5,
        )
        ax.set_title(f"{method}: позначено {int(mask.sum())} точок")
        ax.set_ylabel("ІСЦ, %")
        ax.legend()

    fig.tight_layout()
    fig.savefig(output / "01_anomalies.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for method in METHODS:
        subset = scores.loc[scores["method"] == method]
        best_by_degree = subset.groupby("degree")["cv_rmse"].min()

        axes[0].plot(
            best_by_degree.index,
            best_by_degree.values,
            "o-",
            label=method,
        )

    axes[0].set(
        title="CV: мінімум за варіантами сезонності",
        xlabel="Степінь полінома",
        ylabel="RMSE",
    )
    axes[0].legend()

    axes[1].bar(
        benchmark["method"],
        benchmark["reconstruction_rmse"],
    )
    axes[1].set(
        title="Перевірка на нових штучних ін'єкціях",
        ylabel="RMSE відновлення початкового ряду",
    )

    fig.tight_layout()
    fig.savefig(output / "02_comparison.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))

    for window, group in calibration.groupby("window"):
        group = group.sort_values("q")
        ax.plot(
            group["q"],
            group["reconstruction_rmse"],
            "o-",
            label=f"window={window}",
        )

    ax.set(
        title="R&D: навчання параметрів адаптивного детектора",
        xlabel="Поріг q",
        ylabel="Середній RMSE відновлення",
    )
    ax.legend()

    fig.tight_layout()
    fig.savefig(output / "03_calibration.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(dates, y, color="black", label="Реальні дані")

    for method, predicted in holdout_predictions.items():
        ax.plot(
            dates.iloc[split:],
            predicted,
            "--",
            label=method,
            alpha=0.85,
        )

    ax.axvline(
        dates.iloc[split],
        color="gray",
        linestyle=":",
        label="Початок незалежного test",
    )
    ax.set(
        title="Прогноз на незалежному часовому holdout",
        ylabel="ІСЦ, %",
    )
    ax.legend(ncol=2)

    fig.tight_layout()
    fig.savefig(output / "04_holdout.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(13, 6))

    ax.plot(dates, y, label="Оригінальний ІСЦ", alpha=0.7)
    ax.plot(dates, fitted, label="Навчена модель", linewidth=2)
    ax.plot(
        future_dates,
        forecast,
        "--",
        color="red",
        linewidth=2,
        label=f"Прогноз: {len(forecast)} місяців",
    )
    ax.axvline(dates.iloc[-1], color="gray", linestyle=":")

    ax.set(
        title=(
            f"Фінальна модель: {selected['method']}, "
            f"degree={selected['degree']}, "
            f"seasonal={selected['seasonal']}"
        ),
        ylabel="ІСЦ, %",
    )
    ax.legend()

    fig.tight_layout()
    fig.savefig(output / "05_forecast.png", dpi=180)
    plt.close(fig)


# ============================================================
# Програмна верифікація
# ============================================================

def self_test():
    # 1. МНК відновлює точний квадратичний поліном.
    t = np.arange(60, dtype=float)
    y = 100 + 0.2 * t + 0.01 * t ** 2

    model = fit_lsm(t, y, degree=2, seasonal=False)

    np.testing.assert_allclose(
        predict_lsm(model, t), y, atol=1e-9
    )

    future_t = np.arange(60, 90, dtype=float)
    expected = 100 + 0.2 * future_t + 0.01 * future_t ** 2

    np.testing.assert_allclose(
        predict_lsm(model, future_t),
        expected,
        atol=1e-8,
    )

    # 2. Сталий ряд не змінюється.
    constant = np.full(61, 100.0)

    cleaned, mask, _ = clean_series(
        constant, "adaptive", window=7, q=3
    )

    np.testing.assert_array_equal(cleaned, constant)
    assert not mask.any()

    # 3. Одиночний великий викид виявляється.
    contaminated = constant.copy()
    contaminated[30] = 120.0
    original_copy = contaminated.copy()

    cleaned, mask, _ = clean_series(
        contaminated, "adaptive", window=7, q=3
    )

    assert mask[30]
    np.testing.assert_allclose(cleaned[30], 100.0)

    # Вхідний масив не змінено.
    np.testing.assert_array_equal(contaminated, original_copy)

    # 4. Захист послідовних односпрямованих змін.
    candidates = np.array([False, True, True, True, False])
    residual = np.array([0., 4., 5., 6., 0.])

    protected = protect_persistent_changes(candidates, residual)
    assert not protected.any()

    # 5. Відтворюваність штучних ін'єкцій.
    a, ma = inject_anomalies(
        constant, np.random.default_rng(SEED)
    )
    b, mb = inject_anomalies(
        constant, np.random.default_rng(SEED)
    )

    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(ma, mb)

    # 6. Коректність метрик.
    metrics = regression_metrics(
        np.array([1., 2., 3.]),
        np.array([1., 2., 3.]),
    )

    assert metrics["RMSE"] == 0.0
    assert metrics["R2"] == 1.0


# ============================================================
# Основна програма
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Лабораторна №2: МНК, аномалії, R&D, прогноз."
    )
    parser.add_argument(
        "--input",
        default="results/parsed_data.csv",
    )
    parser.add_argument(
        "--output",
        default="results_lab2",
    )

    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    self_test()

    df = load_series(args.input)
    y = df["cpi"].to_numpy(dtype=float)
    n = len(y)

    # Приблизно 2/3 — development; 1/3 — незалежний test.
    # Довжина test приблизно дорівнює 0.5 довжини development.
    split = int(n * 2 / 3)

    development = y[:split]

    # Детектор навчається тільки на найранішій частині.
    # Саме з цього розміру починається перший CV-fold.
    calibration_end = split // 2
    calibration_data = development[:calibration_end]

    adaptive_parameters, calibration_table = calibrate_detector(
        calibration_data
    )

    # Перевірка детектора на іншому часовому фрагменті
    # та інших реалізаціях доданих аномалій.
    benchmark = injection_benchmark(
        development[calibration_end:],
        adaptive_parameters,
    )

    scores, fold_scores = choose_models(
        development,
        adaptive_parameters,
    )

    selected = config_from_row(scores.iloc[0])

    holdout_table, holdout_predictions = evaluate_holdout(
        y,
        split,
        scores,
        adaptive_parameters,
        selected["method"],
    )

    # Після незалежної оцінки конфігурація вже зафіксована.
    # Перенавчаємо тільки коефіцієнти на всій історії.
    final_model, selected_clean, selected_mask, _ = fit_pipeline(
        y,
        selected,
        adaptive_parameters,
    )

    fitted = predict_lsm(final_model, np.arange(n, dtype=float))

    horizon = math.ceil(0.5 * n)
    future_t = np.arange(n, n + horizon, dtype=float)
    forecast = predict_lsm(final_model, future_t)

    future_dates = pd.date_range(
        df["date"].iloc[-1] + pd.offsets.MonthBegin(1),
        periods=horizon,
        freq="MS",
    )

    # Всі методи зберігаємо для дослідження,
    # навіть якщо CV обрала відсутність очищення.
    cleaned_variants = {}
    masks = {}
    anomaly_scores = {}

    for method in METHODS:
        parameters = method_parameters(
            method, adaptive_parameters
        )

        cleaned, mask, score = clean_series(
            y, method, **parameters
        )

        cleaned_variants[method] = cleaned
        masks[method] = mask
        anomaly_scores[method] = score

    series = df.copy()
    series["development"] = np.arange(n) < split

    for method in ("hampel", "adaptive"):
        series[f"{method}_clean"] = cleaned_variants[method]
        series[f"{method}_flag"] = masks[method]
        series[f"{method}_score"] = anomaly_scores[method]

    series["selected_clean"] = selected_clean
    series["fitted"] = fitted
    series["residual_original"] = y - fitted

    future_table = pd.DataFrame({
        "date": future_dates,
        "forecast_cpi": forecast,
        "forecast_inflation_pct": forecast - 100,
    })

    holdout_series = pd.DataFrame({
        "date": df["date"].iloc[split:].to_numpy(),
        "actual_cpi": y[split:],
    })

    for name, predicted in holdout_predictions.items():
        holdout_series[name] = predicted

    tables = {
        "series": series,
        "forecast": future_table,
        "models_cv": scores,
        "cv_folds": fold_scores,
        "detector_calibration": calibration_table,
        "injection_benchmark": benchmark,
        "holdout_metrics": holdout_table,
        "holdout_predictions": holdout_series,
    }

    for name, table in tables.items():
        save_table(output / f"{name}.csv", table)

    # Перевірка запису та читання прогнозу.
    restored = pd.read_csv(
        output / "forecast.csv",
        parse_dates=["date"],
    )

    np.testing.assert_allclose(
        restored["forecast_cpi"].to_numpy(),
        forecast,
    )

    assert len(forecast) == math.ceil(0.5 * n)
    assert future_dates[0] > df["date"].iloc[-1]
    assert np.isfinite(forecast).all()

    selected_test = holdout_table.loc[
        holdout_table["selected_by_cv"]
    ].iloc[0]

    naive_test = holdout_table.loc[
        holdout_table["method"] == "naive_last"
    ].iloc[0]

    model_info = {
        "input_file": str(Path(args.input)),
        "seed": SEED,
        "n_observations": n,
        "development_n": split,
        "holdout_n": n - split,
        "detector_calibration_n": calibration_end,
        "forecast_horizon": horizon,
        "selection_metric": "RMSE на часовій крос-валідації",
        "selected_pipeline": selected,
        "adaptive_parameters": adaptive_parameters,
        "polynomial_model": final_model,
        "time_definition": "t = номер місяця від початку ряду",
        "polynomial_basis": "z=(t-center)/scale; 1,z,...,z^degree",
        "seasonal_basis": "sin(2*pi*t/12), cos(2*pi*t/12)",
        "flagged_by_selected_method": int(selected_mask.sum()),
        "holdout_used_for_selection": False,
        "forecast_intervals": "Не розраховувалися",
        "self_tests_passed": True,
    }

    save_json(output / "model.json", model_info)

    make_plots(
        df,
        cleaned_variants,
        masks,
        scores,
        calibration_table,
        benchmark,
        split,
        holdout_predictions,
        fitted,
        future_dates,
        forecast,
        selected,
        output,
    )

    comparison = (
        "кращий"
        if selected_test["RMSE"] < naive_test["RMSE"]
        else "не кращий"
    )

    summary = [
        "ЛАБОРАТОРНА РОБОТА №2",
        "Статистичне навчання часових рядів",
        "",
        f"Вхідний файл: {args.input}",
        f"Кількість щомісячних спостережень: {n}",
        f"Development: {split}; незалежний test: {n - split}.",
        f"Калібрування детектора: перші {calibration_end} точок.",
        "",
        "R&D-АЛГОРИТМ:",
        "Двомасштабний детектор на основі локальних медіан і MAD.",
        "Викид підтверджується на двох масштабах і в одному напрямку.",
        "Серії з >=3 односпрямованих кандидатів захищаються від очищення.",
        "Параметри навчаються за RMSE відновлення після штучних ін'єкцій.",
        f"Обрані параметри: {adaptive_parameters}.",
        "",
        "ОБРАНА МОДЕЛЬ:",
        f"Метод підготовки: {selected['method']}.",
        f"Степінь полінома: {selected['degree']}.",
        f"Річна гармоніка: {selected['seasonal']}.",
        f"CV RMSE: {scores.iloc[0]['cv_rmse']:.6f}.",
        f"Test RMSE: {selected_test['RMSE']:.6f}.",
        f"Test MAE: {selected_test['MAE']:.6f}.",
        f"Test R²: {selected_test['R2']}.",
        f"На test прогноз {comparison} за naive_last за RMSE.",
        "",
        f"Горизонт фінального прогнозу: {horizon} місяців.",
        "Горизонт = ceil(0.5 * кількість спостережень).",
        "",
        "ОБМЕЖЕННЯ:",
        "Позначені точки — потенційні аномалії, а не доведені помилки.",
        "Реальні інфляційні шоки можуть бути економічно значущими.",
        "Метрики ін'єкцій оцінюють додані збурення, не справжні аномалії.",
        "Захист послідовних змін може пропустити пакет помилок.",
        "Центровані вікна означають ретроспективне, а не онлайн-очищення.",
        "У CV і holdout очищення виконується тільки всередині train.",
        "Поліноміальна екстраполяція на довгий горизонт може бути нестійкою.",
        "ІСЦ нижче 100% означає зниження цін до попереднього місяця.",
        "Прогноз є навчальним експериментом, не економічною рекомендацією.",
        "",
        "ВЕРИФІКАЦІЯ:",
        "Програмні тести пройдено.",
        "Довжину прогнозу перевірено.",
        "Читання/запис CSV перевірено.",
        "Тестовий період не використовувався для вибору моделі.",
    ]

    if np.any(forecast <= 0):
        summary.append(
            "УВАГА: прогноз містить недопустимий ІСЦ <= 0. "
            "Модель непридатна для такого горизонту."
        )

    text = "\n".join(summary)
    (output / "summary.txt").write_text(text, encoding="utf-8")

    print(text)

    print("\nНАЙКРАЩІ КОНФІГУРАЦІЇ ЗА CV:")
    print(scores.head(10).to_string(index=False))

    print("\nНЕЗАЛЕЖНИЙ TEST:")
    print(holdout_table.to_string(index=False))

    print("\nПЕРЕВІРКА НА ШТУЧНИХ ІН'ЄКЦІЯХ:")
    print(benchmark.to_string(index=False))

    print(f"\nРезультати: {output.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, np.linalg.LinAlgError) as error:
        raise SystemExit(f"Помилка: {error}") from error