"""
ЛР5. Ковзне вікно, ARMA/ARIMA/SARIMA та статистичне навчання.
Оптимізація: expanding-window CV тільки всередині train.
Фінальний test не використовується для налаштування параметрів.
Відсутні записи продажів трактуються як нульові продажі.
"""

import argparse
import json
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, kpss

SEASON = 12
ISSUES = []


def issue(stage, model, error):
    ISSUES.append({
        "stage": stage,
        "model": model,
        "message": str(error),
    })


def load_data(path, sheet):
    raw = pd.read_excel(path, sheet_name=sheet)
    required = ["Order Date", "Region", "Sales"]
    if not set(required).issubset(raw.columns):
        raise ValueError(f"Потрібні колонки: {required}")

    d = raw[required].copy()
    d["Order Date"] = pd.to_datetime(d["Order Date"], errors="coerce")
    d["Sales"] = pd.to_numeric(d["Sales"], errors="coerce")
    if d.isna().any().any() or not np.isfinite(d["Sales"]).all():
        raise ValueError("У датах, регіонах або продажах є некоректні значення.")

    d["Region"] = d["Region"].astype(str).str.strip()
    if d["Region"].eq("").any() or d["Region"].eq("TOTAL").any():
        raise ValueError("Порожній регіон або зарезервована назва TOTAL.")

    d["month"] = d["Order Date"].dt.to_period("M").dt.to_timestamp()
    months = pd.date_range(d.month.min(), d.month.max(), freq="MS")
    monthly = d.groupby(["month", "Region"])["Sales"].sum().unstack()
    missing_cells = int(monthly.reindex(months).isna().sum().sum())
    monthly = monthly.reindex(months).fillna(0.0).astype(float)
    monthly.insert(0, "TOTAL", monthly.sum(axis=1))
    monthly.index.name = "date"

    np.testing.assert_allclose(monthly.TOTAL.sum(), d.Sales.sum())
    np.testing.assert_allclose(
        monthly.TOTAL, monthly.drop(columns="TOTAL").sum(axis=1)
    )
    return monthly, missing_cells


def diagnostics(y):
    a = np.asarray(y, dtype=float)
    result = {
        "n": len(a), "mean": a.mean(), "std": a.std(ddof=1),
        "cv": a.std(ddof=1) / abs(a.mean()) if abs(a.mean()) > 1e-12 else np.nan,
        "acf1": pd.Series(a).autocorr(1),
        "acf12": pd.Series(a).autocorr(SEASON),
        "adf_p": np.nan, "kpss_p": np.nan,
        "seasonal_strength": np.nan, "trend_strength": np.nan,
    }
    if np.ptp(a) < 1e-12:
        return result

    for name, function in [
        ("adf_p", lambda: adfuller(a, autolag="AIC")[1]),
        ("kpss_p", lambda: kpss(a, regression="c", nlags="auto")[1]),
    ]:
        try:
            with warnings.catch_warnings(record=True) as caught:
                result[name] = function()
            for warning in caught:
                issue("diagnostics", name, warning.message)
        except Exception as error:
            issue("diagnostics", name, error)

    if len(a) >= 2 * SEASON:
        stl = STL(a, period=SEASON, robust=True).fit()
        for label, component in [
            ("seasonal_strength", stl.seasonal),
            ("trend_strength", stl.trend),
        ]:
            denominator = np.var(component + stl.resid)
            result[label] = (
                max(0.0, 1 - np.var(stl.resid) / denominator)
                if denominator > 1e-12 else 0.0
            )
    return result


def sma(y, horizon, window):
    values = list(np.asarray(y, dtype=float))
    for _ in range(horizon):
        values.append(float(np.mean(values[-window:])))
    return np.asarray(values[-horizon:])


def features(t):
    t = np.asarray(t, dtype=float)
    return np.column_stack([
        t / SEASON,
        *[(t.astype(int) % SEASON == month).astype(float)
          for month in range(1, SEASON)],
    ])


def predict(y, horizon, family, params):
    y = np.asarray(y, dtype=float)
    scale = max(float(np.std(y)), 1.0)
    z = y / scale
    residual, model_df = None, 0

    with warnings.catch_warnings(record=True) as caught:
        if family == "Naive":
            forecast = np.repeat(y[-1], horizon)
        elif family == "SeasonalNaive":
            forecast = np.resize(y[-SEASON:], horizon)
        elif family == "SMA":
            forecast = sma(y, horizon, params["window"])
        elif family == "Ridge":
            model = make_pipeline(
                StandardScaler(), Ridge(alpha=params["alpha"])
            )
            model.fit(features(np.arange(len(y))), z)
            forecast = model.predict(
                features(np.arange(len(y), len(y) + horizon))
            ) * scale
        elif family == "HoltWinters":
            if len(y) < 2 * SEASON:
                raise ValueError("Holt-Winters потребує щонайменше 24 місяці.")
            model = ExponentialSmoothing(
                z, trend="add", seasonal="add",
                seasonal_periods=SEASON,
                initialization_method="estimated",
            ).fit(optimized=True)
            forecast = np.asarray(model.forecast(horizon)) * scale
        else:
            order = tuple(params["order"])
            seasonal = tuple(params.get("seasonal_order", (0, 0, 0, 0)))
            trend = "c" if order[1] == 0 and seasonal[1] == 0 else "n"
            model = SARIMAX(
                z, order=order, seasonal_order=seasonal, trend=trend,
                enforce_stationarity=True, enforce_invertibility=True,
            ).fit(disp=False, maxiter=200)
            if not model.mle_retvals.get("converged", True):
                raise RuntimeError("Оптимізатор SARIMAX не збігся.")
            forecast = np.asarray(model.forecast(horizon)) * scale
            burn = max(int(model.loglikelihood_burn), SEASON)
            residual = np.asarray(model.resid)[burn:] * scale
            model_df = order[0] + order[2] + seasonal[0] + seasonal[2]

    for warning in caught:
        issue("fit-warning", family, warning.message)
    if len(forecast) != horizon or not np.isfinite(forecast).all():
        raise ValueError("Модель повернула некоректний прогноз.")
    return np.asarray(forecast), residual, model_df


def candidates():
    return {
        "SMA": [{"window": w} for w in (2, 3, 4, 6, 9, 12)],
        "ARMA": [
            {"order": order}
            for order in ((1, 0, 0), (0, 0, 1), (1, 0, 1), (2, 0, 0))
        ],
        "ARIMA": [
            {"order": order}
            for order in ((0, 1, 0), (1, 1, 0), (0, 1, 1), (1, 1, 1))
        ],
        "SARIMA": [
            {"order": (0, 1, 0), "seasonal_order": (0, 0, 1, 12)},
            {"order": (1, 0, 0), "seasonal_order": (1, 0, 0, 12)},
            {"order": (0, 1, 1), "seasonal_order": (0, 1, 1, 12)},
            {"order": (1, 0, 0), "seasonal_order": (0, 1, 0, 12)},
        ],
        "Ridge": [{"alpha": a} for a in (0.0, 0.1, 1.0, 10.0, 100.0)],
    }


def optimize(y, grids, series):
    n = len(y)
    validation_h = min(6, max(3, n // 8))
    origins = sorted(set(
        o for o in (n - 2 * validation_h, n - validation_h) if o >= 18
    ))
    if not origins:
        raise ValueError("Для CV потрібно щонайменше 24 train-спостереження.")

    selected, rows = {}, []
    for family, grid in grids.items():
        options = []
        for params in grid:
            errors, failure = [], ""
            started = time.perf_counter()
            try:
                for origin in origins:
                    actual = np.asarray(y[origin:origin + validation_h])
                    forecast, _, _ = predict(y[:origin], len(actual), family, params)
                    errors.extend(np.abs(actual - forecast))
                score = float(np.mean(errors))
            except Exception as error:
                score, failure = np.inf, str(error)
                issue(f"{series}:CV", family, error)

            rows.append({
                "series": series, "family": family,
                "params": json.dumps(params),
                "cv_MAE": score, "origins": str(origins),
                "validation_h": validation_h,
                "seconds": time.perf_counter() - started,
                "error": failure,
            })
            if np.isfinite(score):
                options.append((score, params))

        if options:
            selected[family] = min(options, key=lambda item: item[0])[1]
    return selected, rows


def metrics(actual, forecast, train):
    actual, forecast, train = map(
        lambda x: np.asarray(x, dtype=float), (actual, forecast, train)
    )
    if actual.shape != forecast.shape or len(actual) == 0:
        raise ValueError("Факт і прогноз повинні мати однакову ненульову довжину.")

    absolute = np.abs(actual - forecast)
    denominator = np.abs(actual) + np.abs(forecast)
    ratios = np.divide(
        2 * absolute, denominator,
        out=np.zeros_like(absolute), where=denominator > 0,
    )
    naive_scale = np.mean(np.abs(train[SEASON:] - train[:-SEASON]))
    return {
        "MAE": absolute.mean(),
        "RMSE": np.sqrt(np.mean((actual - forecast) ** 2)),
        "sMAPE": 100 * ratios.mean(),
        "MASE": absolute.mean() / naive_scale if naive_scale > 1e-12 else np.nan,
    }


def model_specs(selected, grids):
    specs = [
        ("Naive", "Naive", {}),
        ("SeasonalNaive", "SeasonalNaive", {}),
        ("HoltWinters", "HoltWinters", {}),
    ]
    for family, grid in grids.items():
        baseline = {"window": 12} if family == "SMA" else grid[0]
        specs.append((family + "(base)", family, baseline))
        if family in selected:
            specs.append((family + "(opt)", family, selected[family]))
    return specs


def save_plot(y, train, test, forecasts, future, interval, out):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes[0, 0].plot(y.index, y, label="Продажі")
    axes[0, 0].plot(y.index, y.rolling(12).mean(), label="MA(12)")
    axes[0, 0].set_title("TOTAL: ряд та ковзне середнє")
    axes[0, 0].legend()

    lags = min(20, len(train) // 2 - 1)
    if train.std() > 1e-12:
        plot_acf(train, lags=lags, ax=axes[0, 1])
        plot_pacf(train, lags=lags, method="ywm", ax=axes[1, 0])
    axes[0, 1].set_title("ACF: лише train")
    axes[1, 0].set_title("PACF: лише train")

    ax = axes[1, 1]
    ax.plot(train.index, train, color="gray", label="Train")
    ax.plot(test.index, test, color="black", linewidth=2, label="Test")
    for name, forecast in forecasts.items():
        if "(opt)" in name or name == "SeasonalNaive":
            ax.plot(test.index, forecast[:len(test)], label=name)
    ax.set_title("Фіксований початок прогнозування")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "lab5_total.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(y.index, y, color="black", label="Історія")
    for name in future:
        ax.plot(future.index, future[name], label=name)
    for factor in (0.5, 1, 1.5, 2):
        h = int(round(interval * factor))
        ax.axvline(future.index[h - 1], color="gray", linestyle=":", alpha=0.5)
    ax.set_title(f"Екстраполяція; опорний інтервал T={interval} міс.")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "lab5_extrapolation.png", dpi=160)
    plt.close(fig)


def self_test():
    np.testing.assert_allclose(sma([1, 2, 3], 2, 2), [2.5, 2.75])
    result = metrics([0, 2], [0, 2], np.arange(24.0))
    assert result["MAE"] == 0 and result["sMAPE"] == 0
    forecast, _, _ = predict(np.arange(24.0), 6, "Ridge", {"alpha": 1.0})
    assert forecast.shape == (6,) and np.isfinite(forecast).all()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default="Data_Set_11.xlsx")
    parser.add_argument("--sheet", default="Orders")
    parser.add_argument("--train", type=int, default=24)
    parser.add_argument("--interval", type=int, default=None)
    parser.add_argument("--total-only", action="store_true")
    parser.add_argument("--out", default="results_lab5")
    args = parser.parse_args()

    if args.train < 24:
        parser.error("--train має бути не менше 24.")
    interval = args.interval if args.interval is not None else args.train
    if interval < 2 or interval % 2:
        parser.error("--interval має бути додатним парним числом.")

    horizons = [interval // 2, interval, 3 * interval // 2, 2 * interval]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    self_test()
    df, missing = load_data(args.file, args.sheet)

    if len(df) <= args.train:
        raise ValueError("Після train немає даних для незалежного test.")
    print(f"Місяців: {len(df)}; регіонів: {len(df.columns) - 1}")
    print(f"Train={args.train}; T={interval}; горизонти={horizons}")
    print(f"Нульове заповнення відсутніх комбінацій місяць/регіон: {missing}")

    grids = candidates()
    scores, cv_rows, params_rows, props, summaries, forecasts_rows = [], [], [], [], [], []
    total_plot = None

    for column in (["TOTAL"] if args.total_only else df.columns):
        print(f"\nОбробка: {column}", flush=True)
        y = df[column]
        train = y.iloc[:args.train]
        test = y.iloc[args.train:args.train + max(horizons)]
        props.append({"series": column, **diagnostics(train)})
        selected, tuning = optimize(train.to_numpy(), grids, column)
        cv_rows.extend(tuning)
        predictions = {}

        for name, family, params in model_specs(selected, grids):
            started = time.perf_counter()
            try:
                forecast, residual, model_df = predict(
                    train, max(horizons), family, params
                )
                predictions[name] = forecast
                lb_p = np.nan
                if residual is not None and len(residual) > model_df + 3:
                    lag = min(10, len(residual) - 1)
                    if lag > model_df and np.std(residual) > 1e-12:
                        lb_p = acorr_ljungbox(
                            residual, lags=[lag], model_df=model_df
                        )["lb_pvalue"].iloc[0]
                params_rows.append({
                    "series": column, "model": name,
                    "params": json.dumps(params), "residual_lb_p": lb_p,
                    "fit_seconds": time.perf_counter() - started, "status": "OK",
                })
                for h in horizons:
                    available = len(test) >= h
                    row = {
                        "series": column, "model": name, "horizon": h,
                        "status": "OK" if available else "insufficient_test",
                    }
                    if available:
                        row.update(metrics(test.iloc[:h], forecast[:h], train))
                    scores.append(row)
                for i, date in enumerate(test.index):
                    forecasts_rows.append({
                        "series": column, "model": name, "date": date,
                        "actual": test.iloc[i], "forecast": forecast[i],
                    })
            except Exception as error:
                issue(f"{column}:backtest", name, error)
                params_rows.append({
                    "series": column, "model": name,
                    "params": json.dumps(params), "status": str(error),
                })

        valid = pd.DataFrame([r for r in scores if r["series"] == column])
        valid = valid[valid.status == "OK"]
        if not valid.empty:
            ranking = valid.groupby("model")["sMAPE"].mean().sort_values()
            summaries.append({
                "series": column,
                "best_on_test": ranking.index[0],
                "mean_test_sMAPE": ranking.iloc[0],
                "note": "Описовий переможець; не незалежна оцінка після вибору.",
            })
            print(ranking.round(3).to_string())

        if column == "TOTAL":
            future_index = pd.date_range(
                y.index[-1] + pd.offsets.MonthBegin(1),
                periods=max(horizons), freq="MS",
            )
            future = pd.DataFrame(index=future_index)
            # Параметри зафіксовані за train; коефіцієнти переоцінено на всій історії.
            for name, family, params in model_specs(selected, grids):
                if "(opt)" not in name and name not in ("SeasonalNaive", "HoltWinters"):
                    continue
                try:
                    future[name] = predict(y, max(horizons), family, params)[0]
                except Exception as error:
                    issue("TOTAL:future", name, error)
            future.index.name = "date"
            total_plot = (y, train, test, predictions, future)
            future.to_csv(out / "future_total.csv", encoding="utf-8-sig")

    score_df = pd.DataFrame(scores)
    comparisons = []
    for series in score_df.series.unique():
        for family in grids:
            base = score_df[(score_df.series == series)
                            & (score_df.model == family + "(base)")]
            opt = score_df[(score_df.series == series)
                           & (score_df.model == family + "(opt)")]
            for _, row in base.merge(opt, on="horizon", suffixes=("_base", "_opt")).iterrows():
                if row.status_base != "OK" or row.status_opt != "OK":
                    continue
                b, o = row.MAE_base, row.MAE_opt
                comparisons.append({
                    "series": series, "family": family, "horizon": row.horizon,
                    "MAE_base": b, "MAE_opt": o,
                    "MAE_improvement_pct": 100 * (b - o) / b if b > 0 else np.nan,
                })

    tables = {
        "metrics": score_df, "cv": pd.DataFrame(cv_rows),
        "parameters": pd.DataFrame(params_rows), "properties_train": pd.DataFrame(props),
        "summary": pd.DataFrame(summaries), "optimization": pd.DataFrame(comparisons),
        "backtest": pd.DataFrame(forecasts_rows),
        "issues": pd.DataFrame(ISSUES, columns=["stage", "model", "message"]),
    }
    df.to_csv(out / "monthly_sales.csv", encoding="utf-8-sig")
    df.iloc[:args.train].corr().to_csv(out / "train_correlations.csv", encoding="utf-8-sig")
    with pd.ExcelWriter(out / "lab5_results.xlsx") as writer:
        for name, table in tables.items():
            table.to_csv(out / f"{name}.csv", index=False, encoding="utf-8-sig")
            table.to_excel(writer, sheet_name=name, index=False)

    if total_plot:
        save_plot(*total_plot, interval, out)
        y = total_plot[0]
        decomposition = STL(y, period=SEASON, robust=True).fit()
        fig = decomposition.plot()
        fig.set_size_inches(12, 8)
        fig.tight_layout()
        fig.savefig(out / "total_decomposition.png", dpi=160)
        plt.close(fig)

    notes = [
        "Параметри оптимізовано тільки на навчальній частині.",
        "CV використовує короткі послідовні validation-відрізки.",
        "Лише 24 train-місяці: сезонні моделі та тести мають обмежену надійність.",
        "Немає фактичних спостережень — немає метрик відповідного горизонту.",
        "Середнє sMAPE за горизонтами використовує вкладені, не незалежні відрізки.",
        "Holt-Winters — класична модель згладжування; Ridge — регресійне навчання.",
        "SMA та MA-компонента ARMA — різні поняття.",
        "Нульове заповнення допустиме лише якщо відсутність замовлень означає нуль продажів.",
        "Негативні прогнози не обрізаються приховано; їх треба врахувати в аналізі.",
        "Ljung-Box p>0.05 означає невідхилення нульової гіпотези, не доказ адекватності.",
        "Оптимізація може погіршити test: див. optimization.csv, від'ємний приріст.",
        "Рекомендації: обирати складність за CV, порівнювати з SeasonalNaive;",
        "за короткої історії надавати перевагу стабільним простим моделям;",
        "за сезонністю порівнювати SARIMA/Holt-Winters/Ridge, а не покладатися лише на ACF.",
        "Окремі прогнози TOTAL та регіонів не узгоджені ієрархічно.",
    ]
    (out / "methodology.txt").write_text("\n".join(notes), encoding="utf-8")
    print("\nПідсумок:\n", tables["summary"].to_string(index=False))
    print(f"\nГотово. Результати: {out.resolve()}")


if __name__ == "__main__":
    main()