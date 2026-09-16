"""
Лабораторна №1: Отримання та підготовка Time Series. ІІІ рівень.
Дані: щомісячний індекс споживчих цін України, % до попереднього місяця.
Модель: поліноміальний тренд + сезонність + AR(1) із bootstrap інновацій.

Встановлення:
    pip install numpy pandas matplotlib requests beautifulsoup4 openpyxl

Запуск:
    python main.py
    python main.py --start 2015 --end 2025
    python main.py --offline

Результати зберігаються в results/.
Верифікація на зібраних даних не є доказом точності майбутніх прогнозів.
"""

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

URL = "https://index.minfin.com.ua/ua/economy/index/inflation/"
OUT = Path("results")
SEED = 42


def save_json(path, data):
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def parse_html(html):
    records = []
    soup = BeautifulSoup(html, "html.parser")
    for table in soup.find_all("table"):
        for row in table.find_all("tr"):
            cells = row.find_all(["td", "th"])
            if len(cells) < 13:
                continue
            year_text = cells[0].get_text(" ", strip=True)
            if not re.fullmatch(r"(?:19|20)\d{2}", year_text):
                continue
            year = int(year_text)
            for month, cell in enumerate(cells[1:13], 1):
                text = re.sub(r"\s+", "", cell.get_text()).replace(",", ".")
                text = text.rstrip("%")
                if re.fullmatch(r"\d+(?:\.\d+)?", text):
                    records.append({
                        "date": pd.Timestamp(year, month, 1),
                        "cpi": float(text),
                    })
    if not records:
        raise ValueError("Таблицю інфляції не знайдено: перевірте структуру сайту.")
    df = pd.DataFrame(records)
    if df.groupby("date")["cpi"].nunique().gt(1).any():
        raise ValueError("На сайті знайдено суперечливі значення для однієї дати.")
    return df.drop_duplicates("date").sort_values("date").reset_index(drop=True)


def load_data(start, end, offline):
    cache = OUT / "source.html"
    if offline:
        if not cache.exists():
            raise FileNotFoundError("Немає source.html. Спочатку запустіть онлайн.")
        html = cache.read_text(encoding="utf-8")
    else:
        retry = Retry(
            total=3, backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
        )
        with requests.Session() as session:
            session.mount("https://", HTTPAdapter(max_retries=retry))
            response = session.get(
                URL, headers={"User-Agent": "TimeSeriesLab/1.0"},
                timeout=(10, 40),
            )
            response.raise_for_status()
            response.encoding = "utf-8"
            html = response.text

    parsed = parse_html(html)
    df = parsed.loc[parsed.date.dt.year.between(start, end)].copy()
    expected = pd.date_range(f"{start}-01-01", f"{end}-12-01", freq="MS")
    missing = expected.difference(pd.DatetimeIndex(df.date))
    if len(missing):
        dates = ", ".join(missing.strftime("%Y-%m"))
        raise ValueError(f"Відсутні місячні дані: {dates}. Змініть період.")
    if len(df) < 48 or not np.isfinite(df.cpi).all() or (df.cpi <= 0).any():
        raise ValueError("Потрібно не менше 48 коректних місячних спостережень.")

    df = df.reset_index(drop=True)
    if not offline:
        cache.write_text(html, encoding="utf-8")
        save_json(OUT / "source_metadata.json", {
            "url": URL,
            "downloaded_at": datetime.now(timezone.utc).isoformat(),
        })
    df["inflation_pct"] = df.cpi - 100
    df.to_csv(OUT / "parsed_data.csv", index=False, encoding="utf-8-sig")
    restored = pd.read_csv(OUT / "parsed_data.csv", parse_dates=["date"])
    pd.testing.assert_frame_equal(df, restored, check_dtype=False)
    return df


def describe(x):
    x = np.asarray(x, dtype=float)
    centered = x - x.mean()
    std = x.std(ddof=0)
    return {
        "n": len(x),
        "mean": float(x.mean()),
        "median": float(np.median(x)),
        "variance_ddof0": float(x.var(ddof=0)),
        "variance_ddof1": float(x.var(ddof=1)),
        "std_ddof0": float(std),
        "min": float(x.min()),
        "max": float(x.max()),
        "skewness": float(np.mean(centered**3) / std**3) if std else 0.0,
        "excess_kurtosis": float(np.mean(centered**4) / std**4 - 3) if std else 0.0,
    }


def acf(x, lag):
    x = np.asarray(x) - np.mean(x)
    denominator = x @ x
    return float(x[:-lag] @ x[lag:] / denominator) if denominator else 0.0


def design(t, degree, seasonal):
    columns = [t**power for power in range(degree + 1)]
    if seasonal:
        columns += [np.sin(2 * np.pi * t), np.cos(2 * np.pi * t)]
    return np.column_stack(columns)


def choose_model(t, y):
    candidates = []
    n = len(y)
    for degree in range(4):
        for seasonal in (False, True):
            X = design(t, degree, seasonal)
            errors = []
            for fraction in (0.60, 0.75, 0.90):
                stop = int(n * fraction)
                finish = min(n, stop + max(3, n // 10))
                beta = np.linalg.lstsq(X[:stop], y[:stop], rcond=None)[0]
                errors.extend((y[stop:finish] - X[stop:finish] @ beta) ** 2)
            candidates.append({
                "degree": degree,
                "seasonal": seasonal,
                "cv_rmse": float(np.sqrt(np.mean(errors))),
            })
    scores = pd.DataFrame(candidates).sort_values(
        ["cv_rmse", "degree", "seasonal"]
    ).reset_index(drop=True)
    best = scores.iloc[0]
    return int(best.degree), bool(best.seasonal), scores


def fit_model(t, y, degree, seasonal):
    X = design(t, degree, seasonal)
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    fitted = X @ beta
    residual = y - fitted
    denominator = residual[:-1] @ residual[:-1]
    raw_phi = float(residual[:-1] @ residual[1:] / denominator) if denominator else 0
    phi = float(np.clip(raw_phi, -0.98, 0.98))
    innovations = residual[1:] - phi * residual[:-1]
    innovations -= innovations.mean()
    return X, beta, fitted, residual, phi, raw_phi, innovations


def simulate(fitted, phi, innovations, rng):
    burn = 500
    shocks = rng.choice(innovations, size=len(fitted) + burn, replace=True)
    residual = np.zeros(len(shocks))
    for i in range(1, len(shocks)):
        residual[i] = phi * residual[i - 1] + shocks[i]
    return fitted + residual[burn:]


def metrics(y, X, t):
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    residual = y - X @ beta
    return np.array([
        y.mean(), y.var(), residual.var(),
        acf(residual, 1), acf(residual, 12),
        np.polyfit(t, y, 1)[0],
    ])


def verify(y, simulated, X, t, fitted, phi, innovations, repeats):
    names = [
        "Середнє", "Дисперсія ряду", "Дисперсія залишків",
        "ACF залишків, lag=1", "ACF залишків, lag=12",
        "Лінійний нахил, в.п./рік",
    ]
    rng = np.random.default_rng(SEED + 1)
    ensemble = np.array([
        metrics(simulate(fitted, phi, innovations, rng), X, t)
        for _ in range(repeats)
    ])
    low, high = np.quantile(ensemble, [0.025, 0.975], axis=0)
    actual = metrics(y, X, t)
    model = metrics(simulated, X, t)
    table = pd.DataFrame({
        "metric": names,
        "real": actual,
        "synthetic": model,
        "absolute_difference": np.abs(actual - model),
        "simulation_q025": low,
        "simulation_q975": high,
        "real_inside_interval": (actual >= low) & (actual <= high),
    })
    return table


def plots(df, t, y, synthetic, fitted, trend, residual, synthetic_residual, innovations):
    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.25})
    fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True)
    axes[0].plot(df.date, y, label="Реальні дані", alpha=0.8)
    axes[0].plot(df.date, fitted, label="Тренд + сезонність", linewidth=2)
    axes[0].plot(df.date, trend, "--", label="Поліноміальний тренд")
    axes[0].set_title("Індекс споживчих цін України")
    axes[0].set_ylabel("% до попереднього місяця")
    axes[1].plot(df.date, y, label="Реальні дані", alpha=0.65)
    axes[1].plot(df.date, synthetic, label="Синтетичні дані", alpha=0.7)
    axes[1].set_ylabel("ІСЦ, %")
    axes[2].plot(df.date, residual, label="Залишки реального ряду")
    axes[2].plot(df.date, synthetic_residual, label="Залишки моделі", alpha=0.65)
    axes[2].axhline(0, color="black", linewidth=0.7)
    axes[2].set_ylabel("Відхилення, в.п.")
    for ax in axes:
        ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "time_series.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, a, b, title in [
        (axes[0, 0], y, synthetic, "Розподіл значень ІСЦ"),
        (axes[0, 1], residual, synthetic_residual, "Розподіл залишків"),
    ]:
        bins = np.histogram_bin_edges(np.r_[a, b], bins="auto")
        ax.hist(a, bins=bins, density=True, alpha=0.55, label="Реальні")
        ax.hist(b, bins=bins, density=True, alpha=0.55, label="Синтетичні")
        ax.set(title=title, ylabel="Щільність")
        ax.legend()
    lags = np.arange(1, min(25, len(y) // 4))
    axes[1, 0].plot(lags, [acf(residual, k) for k in lags], "o-", label="Реальні")
    axes[1, 0].plot(lags, [acf(synthetic_residual, k) for k in lags], "s-", label="Модель")
    axes[1, 0].set(title="Автокореляція залишків", xlabel="Лаг, місяців", ylabel="ACF")
    axes[1, 0].legend()
    axes[1, 1].hist(innovations, bins="auto", density=True, alpha=0.75)
    axes[1, 1].set(title="Емпіричний розподіл інновацій AR(1)", ylabel="Щільність")
    fig.tight_layout()
    fig.savefig(OUT / "distributions_acf.png", dpi=180)
    plt.close(fig)


def self_test():
    x = np.array([1., 2., 3., 4.])
    np.testing.assert_allclose(describe(x)["mean"], 2.5)
    np.testing.assert_allclose(describe(x)["variance_ddof0"], 1.25)
    t = np.linspace(0, 3, 60)
    X = design(t, 2, False)
    beta = np.array([3., 2., 0.5])
    np.testing.assert_allclose(np.linalg.lstsq(X, X @ beta, rcond=None)[0], beta)
    cells = "<td>100,5</td>" * 12
    sample = f"<table><tr><td>2020</td>{cells}<td>106.2</td></tr></table>"
    parsed = parse_html(sample)
    assert len(parsed) == 12 and parsed.date.iloc[-1].month == 12
    np.testing.assert_allclose(parsed.cpi, 100.5)
    a = simulate(np.zeros(100), 0.5, x - x.mean(), np.random.default_rng(SEED))
    b = simulate(np.zeros(100), 0.5, x - x.mean(), np.random.default_rng(SEED))
    np.testing.assert_array_equal(a, b)


def main():
    parser = argparse.ArgumentParser(description="Time Series, лабораторна №1, ІІІ рівень")
    parser.add_argument("--start", type=int, default=2010)
    parser.add_argument("--end", type=int, default=datetime.now().year - 1)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--simulations", type=int, default=500)
    args = parser.parse_args()
    if args.start > args.end or args.simulations < 100:
        parser.error("Початок має бути ≤ кінця; кількість симуляцій — не менше 100.")

    OUT.mkdir(exist_ok=True)
    self_test()
    df = load_data(args.start, args.end, args.offline)
    y = df.cpi.to_numpy(dtype=float)
    t = np.arange(len(y)) / 12
    degree, seasonal, scores = choose_model(t, y)
    X, beta, fitted, residual, phi, raw_phi, innovations = fit_model(t, y, degree, seasonal)
    synthetic = simulate(fitted, phi, innovations, np.random.default_rng(SEED))
    synthetic_beta = np.linalg.lstsq(X, synthetic, rcond=None)[0]
    synthetic_residual = synthetic - X @ synthetic_beta
    trend = design(t, degree, False) @ beta[:degree + 1]
    check = verify(y, synthetic, X, t, fitted, phi, innovations, args.simulations)

    statistics = pd.DataFrame({
        "real": describe(y), "synthetic": describe(synthetic),
        "real_residual": describe(residual),
        "synthetic_residual": describe(synthetic_residual),
        "innovations": describe(innovations),
    }).rename_axis("statistic").reset_index()
    result = df.assign(
        t_years=t, trend=trend, fitted=fitted,
        residual=residual, synthetic=synthetic,
        synthetic_residual=synthetic_residual,
    )
    tables = {"series": result, "statistics": statistics, "models_cv": scores, "verification": check}
    for name, table in tables.items():
        table.to_csv(OUT / f"{name}.csv", index=False, encoding="utf-8-sig")
    result.to_json(OUT / "series.json", orient="records", date_format="iso", indent=2)
    with pd.ExcelWriter(OUT / "analysis.xlsx") as writer:
        for name, table in tables.items():
            table.to_excel(writer, sheet_name=name, index=False)

    slope = float(np.polyfit(t, y, 1)[0])
    ss_total = float(np.sum((y - y.mean()) ** 2))
    r2 = float(1 - np.sum(residual**2) / ss_total) if ss_total else None
    save_json(OUT / "model.json", {
        "source": URL, "period": [str(df.date.min().date()), str(df.date.max().date())],
        "seed": SEED, "degree": degree, "seasonal": seasonal,
        "time_unit": "роки від початку вибірки; t = номер місяця / 12",
        "basis": "1, t, ..., t^degree; за сезонності: sin(2*pi*t), cos(2*pi*t)",
        "coefficients": beta.tolist(), "phi": phi, "phi_unrestricted": raw_phi,
        "model": "y*(t) = X(t)b + e(t); e(t) = phi*e(t-1) + u*(t)",
        "innovation_distribution": "bootstrap із центрованих емпіричних інновацій",
        "innovation_mean": float(innovations.mean()),
        "innovation_variance": float(innovations.var()),
        "stationary_residual_variance": float(innovations.var() / (1 - phi**2)),
        "trend_r_squared": r2, "simulations": args.simulations,
    })
    plots(df, t, y, synthetic, fitted, trend, residual, synthetic_residual, innovations)

    passed = int(check.real_inside_interval.sum())
    direction = "зростання" if slope > 0 else "зниження" if slope < 0 else "сталість"
    summary = [
        f"Джерело: {URL}",
        f"Період: {args.start}–{args.end}; спостережень: {len(y)}; пропусків: 0.",
        "ІСЦ = 100% означає відсутність зміни цін відносно попереднього місяця.",
        f"Обрано тренд степеня {degree}; річна сезонність: {seasonal}.",
        f"Вибір за часовою крос-валідацією; RMSE: {scores.cv_rmse.iloc[0]:.4f}.",
        f"Загальна лінійна тенденція ІСЦ: {direction}, {slope:.4f} в.п./рік.",
        f"Зміна поліноміального тренду за період: {trend[-1] - trend[0]:.4f} в.п.",
        f"AR(1): phi={phi:.4f}; ACF інновацій lag=1: {acf(innovations, 1):.4f}.",
        f"У 95% симуляційні інтервали потрапило {passed}/{len(check)} характеристик.",
        "Це внутрішня перевірка сумісності, а не незалежний статистичний доказ.",
        "Синтетичний ряд — одна фіксована реалізація, без підгонки її моментів.",
        "Падіння ІСЦ не обов'язково означає падіння цін: ціни падають за ІСЦ < 100%.",
        "AR(1) може не відтворити структурні злами, довгу пам'ять і зміну дисперсії.",
        "За невідповідності характеристик модель потребує уточнення.",
        "Програмні тести та перевірка читання/запису CSV: успішно.",
    ]
    if phi != raw_phi:
        summary.append("Оцінку phi обмежено для забезпечення стаціонарності AR(1).")
    text = "\n".join(summary)
    (OUT / "analysis.txt").write_text(text, encoding="utf-8")
    print(text)
    print("\nСТАТИСТИЧНІ ХАРАКТЕРИСТИКИ\n", statistics.round(5).to_string(index=False))
    print("\nВЕРИФІКАЦІЯ МОДЕЛІ\n", check.round(5).to_string(index=False))
    print(f"\nФайли збережено: {OUT.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except (requests.RequestException, ValueError, OSError) as error:
        raise SystemExit(f"Помилка: {error}") from error