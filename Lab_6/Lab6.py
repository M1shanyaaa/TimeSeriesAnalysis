"""
ЛР6: експоненціальне згладжування, регресія та адаптивний вибір.
pip install numpy pandas matplotlib statsmodels scikit-learn openpyxl

Дані: Orders; Order Date, Region, Sales.
Відсутні комбінації місяць/регіон трактуються як нуль продажів.
Адаптивний алгоритм: часовий CV + властивості train + перевага простоти.
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
from sklearn.linear_model import Ridge, HuberRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import acorr_ljungbox

PERIOD = 12
ISSUES = []


def log_issue(stage, model, message):
    ISSUES.append({"stage": stage, "model": model, "message": str(message)})


def load_data(path, sheet):
    raw = pd.read_excel(path, sheet_name=sheet)
    required = ["Order Date", "Region", "Sales"]
    if not set(required).issubset(raw.columns):
        raise ValueError(f"Потрібні колонки: {required}")
    d = raw[required].copy()
    d["Order Date"] = pd.to_datetime(d["Order Date"], errors="coerce")
    d["Sales"] = pd.to_numeric(d["Sales"], errors="coerce")
    if d.isna().any().any() or not np.isfinite(d.Sales).all():
        raise ValueError("Некоректні дати, регіони або продажі.")
    d["Region"] = d.Region.astype(str).str.strip()
    if d.Region.eq("").any() or d.Region.eq("TOTAL").any():
        raise ValueError("Порожній регіон або зарезервована назва TOTAL.")
    d["month"] = d["Order Date"].dt.to_period("M").dt.to_timestamp()
    months = pd.date_range(d.month.min(), d.month.max(), freq="MS")
    df = d.groupby(["month", "Region"]).Sales.sum().unstack().reindex(months)
    missing = int(df.isna().sum().sum())
    df = df.fillna(0.0).astype(float)
    df.insert(0, "TOTAL", df.sum(axis=1))
    df.index.name = "date"
    np.testing.assert_allclose(df.TOTAL.sum(), d.Sales.sum())
    np.testing.assert_allclose(df.TOTAL, df.drop(columns="TOTAL").sum(axis=1))
    return df, missing


def properties(y):
    y = np.asarray(y, dtype=float)
    std = float(np.std(y))
    result = dict(n=len(y), mean=float(y.mean()), std=std,
                  variance=float(np.var(y)), min=float(y.min()),
                  max=float(y.max()), adf_p=np.nan, kpss_p=np.nan,
                  season_strength=0.0, trend_strength=0.0, instability=0.0)
    result["cv"] = std / abs(y.mean()) if abs(y.mean()) > 1e-12 else np.nan
    result["acf1"] = pd.Series(y).autocorr(1) if std > 1e-12 else np.nan
    result["acf12"] = pd.Series(y).autocorr(PERIOD) if std > 1e-12 else np.nan
    if std < 1e-12:
        return result
    stl = STL(y, period=PERIOD, robust=True).fit()
    for key, component in [("season_strength", stl.seasonal),
                           ("trend_strength", stl.trend)]:
        denominator = np.var(component + stl.resid)
        result[key] = max(0.0, 1 - np.var(stl.resid) / denominator) if denominator else 0
    half = len(y) // 2
    slopes = [np.polyfit(np.arange(len(v)), v, 1)[0]
              for v in (stl.trend[:half], stl.trend[half:])]
    result["instability"] = float(abs(slopes[1] - slopes[0]) * PERIOD / std)
    for key, function in [
        ("adf_p", lambda: adfuller(y, autolag="AIC")[1]),
        ("kpss_p", lambda: kpss(y, regression="c", nlags="auto")[1]),
    ]:
        try:
            with warnings.catch_warnings(record=True) as caught:
                result[key] = float(function())
            for warning in caught:
                log_issue("properties", key, warning.message)
        except Exception as error:
            log_issue("properties", key, error)
    return result


def features(t, degree, harmonics):
    t = np.asarray(t, dtype=float)
    columns = [(t / PERIOD) ** p for p in range(1, degree + 1)]
    for k in range(1, harmonics + 1):
        columns += [np.sin(2 * np.pi * k * t / PERIOD),
                    np.cos(2 * np.pi * k * t / PERIOD)]
    return np.column_stack(columns)


def candidates():
    models = [
        dict(name="Naive", family="B", complexity=1),
        dict(name="SeasonalNaive", family="B", complexity=PERIOD),
        dict(name="SES_fixed", family="E", complexity=1, alpha=0.2),
        dict(name="SES_opt", family="E", complexity=1),
        dict(name="Holt", family="E", complexity=2, trend="add"),
        dict(name="Holt_damped", family="E", complexity=3, trend="add", damped=True),
        dict(name="HW_add", family="E", complexity=14, trend="add", seasonal="add"),
        dict(name="HW_damped", family="E", complexity=15,
             trend="add", seasonal="add", damped=True),
        dict(name="HW_mul", family="E", complexity=14, trend="add", seasonal="mul"),
    ]
    for degree in (1, 2):
        for harmonics in (0, 2, 4):
            for alpha in (0.1, 1.0, 10.0):
                models.append(dict(
                    name=f"Ridge_d{degree}_k{harmonics}_a{alpha}",
                    family="R", complexity=1 + degree + 2 * harmonics,
                    degree=degree, harmonics=harmonics, alpha=alpha))
    models.append(dict(name="Huber_season", family="R", complexity=6,
                       degree=1, harmonics=2, robust=True, alpha=0.1))
    return models


def fit_predict(y, horizon, config):
    y = np.asarray(y, dtype=float)
    n, family = len(y), config["family"]
    scale = max(float(np.std(y)), 1.0)
    z = y / scale
    with warnings.catch_warnings(record=True) as caught:
        if family == "B":
            lag = PERIOD if config["name"] == "SeasonalNaive" else 1
            forecast = np.resize(y[-lag:], horizon)
            fitted = np.r_[np.full(lag, np.nan), y[:-lag]]
        elif family == "E":
            if config.get("seasonal") == "mul" and np.any(y <= 0):
                raise ValueError("Мультиплікативна сезонність потребує додатних даних.")
            model = ExponentialSmoothing(
                z, trend=config.get("trend"), seasonal=config.get("seasonal"),
                damped_trend=config.get("damped", False),
                seasonal_periods=PERIOD if config.get("seasonal") else None,
                initialization_method="estimated")
            options = {"smoothing_level": config["alpha"]} if "alpha" in config else {}
            result = model.fit(optimized=True, **options)
            convergence = getattr(result, "mle_retvals", None)
            if convergence is not None and not convergence.get("success", True):
                raise RuntimeError("Оптимізатор експоненціальної моделі не збігся.")
            forecast = np.asarray(result.forecast(horizon)) * scale
            fitted = np.asarray(result.fittedvalues) * scale
        else:
            estimator = (HuberRegressor(alpha=config["alpha"], max_iter=1000)
                         if config.get("robust") else Ridge(alpha=config["alpha"]))
            model = make_pipeline(StandardScaler(), estimator)
            X = features(np.arange(n + horizon), config["degree"], config["harmonics"])
            model.fit(X[:n], z)
            fitted = model.predict(X[:n]) * scale
            forecast = model.predict(X[n:]) * scale
        for warning in caught:
            log_issue("fit", config["name"], warning.message)
    if len(forecast) != horizon or not np.isfinite(forecast).all():
        raise ValueError("Некоректний прогноз.")
    return fitted, forecast


def metrics(actual, forecast, reference):
    a, f, r = [np.asarray(v, dtype=float) for v in (actual, forecast, reference)]
    if a.shape != f.shape or len(a) == 0:
        raise ValueError("Розміри факту та прогнозу не збігаються.")
    absolute = np.abs(a - f)
    denominator = np.abs(a) + np.abs(f)
    ratio = np.divide(2 * absolute, denominator, out=np.zeros_like(a),
                      where=denominator > 0)
    seasonal_scale = np.mean(np.abs(r[PERIOD:] - r[:-PERIOD]))
    sst = np.sum((a - a.mean()) ** 2)
    return dict(MAE=float(absolute.mean()), RMSE=float(np.sqrt(np.mean((a - f) ** 2))),
                sMAPE=float(100 * ratio.mean()),
                MASE=float(absolute.mean() / seasonal_scale) if seasonal_scale > 1e-12 else np.nan,
                R2=float(1 - np.sum((a - f) ** 2) / sst) if sst > 1e-12 else np.nan)


def cross_validate(y, models, series):
    n = len(y)
    h = min(6, (n - 24) // 2)
    if h < 1:
        raise ValueError("Для CV із сезонними моделями потрібно мінімум 26 train-місяців.")
    origins = sorted(set([24, n - 2 * h, n - h]))
    rows = []
    for config in models:
        losses, error = [], ""
        started = time.perf_counter()
        try:
            for origin in origins:
                actual = y[origin:origin + h]
                _, forecast = fit_predict(y[:origin], h, config)
                losses.append(float(np.mean(np.abs(actual - forecast))))
        except Exception as exception:
            error = str(exception)
            log_issue(f"{series}:CV", config["name"], exception)
        rows.append(dict(
            series=series, model=config["name"], family=config["family"],
            complexity=config["complexity"], cv_MAE=np.mean(losses) if not error else np.nan,
            cv_std=np.std(losses) if not error else np.nan, folds=len(losses),
            origins=str(origins), cv_horizon=h, seconds=time.perf_counter() - started,
            status=error or "OK", params=json.dumps(config)))
    return pd.DataFrame(rows)


def choose(cv, prop):
    valid = cv[(cv.status == "OK") & cv.family.isin(["E", "R"])].copy()
    if valid.empty:
        raise ValueError("Немає успішних експоненціальних або регресійних моделей.")
    group_best = {g: part.sort_values("cv_MAE").iloc[0].model
                  for g, part in valid.groupby("family")}
    best = valid.sort_values("cv_MAE").iloc[0]
    tolerance = 0.05 * max(float(best.cv_MAE), 1e-9)
    pool = valid[valid.cv_MAE <= best.cv_MAE + tolerance]
    preferred = "E" if prop["instability"] > 0.5 or prop["season_strength"] > 0.6 else "R"
    filtered = pool[pool.family == preferred]
    if not filtered.empty:
        pool = filtered
    selected = pool.sort_values(["complexity", "cv_MAE"]).iloc[0]
    reason = (f"season={prop['season_strength']:.3f}; instability={prop['instability']:.3f}; "
              f"перевага {preferred}; серед моделей у межах 5% CV-MAE обрано простішу.")
    return group_best, best.model, selected.model, reason


def draw(series, y, train, forecasts, fits, future, cv, out):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes[0, 0].plot(train.index, train, "k", label="Train")
    for name, values in fits.items():
        axes[0, 0].plot(train.index, values, alpha=0.75, label=name)
    axes[0, 0].set_title("Обробка навчальної вибірки")
    test = y.iloc[len(train):]
    axes[0, 1].plot(test.index, test, "ko-", label="Факт")
    for name, values in forecasts.items():
        axes[0, 1].plot(test.index, values, label=name)
    axes[0, 1].set_title("Незалежний test: вибір моделей уже зафіксований")
    axes[1, 0].plot(y.index, y, "k", label="Історія")
    for name in future:
        axes[1, 0].plot(future.index, future[name], label=name)
    axes[1, 0].set_title("Екстраполяція після переоцінювання на всій історії")
    top = cv[cv.status == "OK"].nsmallest(10, "cv_MAE").iloc[::-1]
    axes[1, 1].barh(top.model, top.cv_MAE)
    axes[1, 1].set_title("Найкращі CV-MAE, лише train")
    for ax in axes.flat:
        ax.grid(alpha=0.25)
    for ax in axes.flat[:3]:
        ax.legend(fontsize=7)
    fig.suptitle(series)
    fig.tight_layout()
    fig.savefig(out / "comparison_total.png", dpi=170)
    plt.close(fig)
    fig = STL(train, period=PERIOD, robust=True).fit().plot()
    fig.set_size_inches(12, 8)
    fig.tight_layout()
    fig.savefig(out / "decomposition_train.png", dpi=170)
    plt.close(fig)


def self_test():
    result = metrics([0., 2.], [0., 2.], np.arange(36.))
    assert result["MAE"] == 0 and result["sMAPE"] == 0
    linear = dict(name="test", family="R", degree=1, harmonics=0, alpha=0.0)
    y = 5 + 2 * np.arange(36.)
    fitted, forecast = fit_predict(y, 6, linear)
    np.testing.assert_allclose(fitted, y, atol=1e-8)
    np.testing.assert_allclose(forecast, 5 + 2 * np.arange(36, 42), atol=1e-8)
    _, seasonal = fit_predict(np.arange(36.), 12, candidates()[1])
    np.testing.assert_array_equal(seasonal, np.arange(24., 36.))


def main():
    parser = argparse.ArgumentParser(description="ЛР6: усі три групи вимог")
    parser.add_argument("--file", default="Data_Set_11.xlsx")
    parser.add_argument("--sheet", default="Orders")
    parser.add_argument("--test", type=int, default=12)
    parser.add_argument("--future", type=int, default=24)
    parser.add_argument("--total-only", action="store_true")
    parser.add_argument("--out", default="results_lab6")
    args = parser.parse_args()
    if args.test < 1 or args.future < 1:
        parser.error("--test і --future мають бути додатними.")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    self_test()
    df, missing = load_data(args.file, args.sheet)
    if len(df) - args.test < 26:
        raise ValueError(f"Є {len(df)} місяців: потрібно train ≥ 26 плюс test={args.test}.")
    models = candidates()
    lookup = {model["name"]: model for model in models}
    all_cv, properties_rows, summary, kpis, predictions, future_rows, fit_rows = [], [], [], [], [], [], []
    for series in (["TOTAL"] if args.total_only else df.columns):
        print(f"\nОбробка: {series}", flush=True)
        y = df[series]
        train, test = y.iloc[:-args.test], y.iloc[-args.test:]
        prop = properties(train)
        properties_rows.append({"series": series, **prop})
        cv = cross_validate(train.to_numpy(), models, series)
        all_cv.append(cv)
        champions, cv_best, adaptive, reason = choose(cv, prop)
        selected = dict(Naive="Naive", SeasonalNaive="SeasonalNaive",
                        E_base="SES_fixed", R_base="Ridge_d1_k0_a1.0",
                        **{f"{g}_best": name for g, name in champions.items()},
                        CV_best=cv_best, Adaptive=adaptive)
        cache, forecasts, fits = {}, {}, {}
        future_index = pd.date_range(y.index[-1] + pd.offsets.MonthBegin(),
                                     periods=args.future, freq="MS")
        future = pd.DataFrame(index=future_index)
        for label, name in selected.items():
            try:
                started = time.perf_counter()
                if name not in cache:
                    cache[name] = fit_predict(train, len(test), lookup[name])
                fitted, forecast = cache[name]
                forecasts[label], fits[label] = forecast, fitted
                for h in sorted(set([min(3, len(test)), min(6, len(test)), len(test)])):
                    kpis.append(dict(series=series, method=label, model=name,
                                     horizon=h, **metrics(test.iloc[:h], forecast[:h], train)))
                valid = np.isfinite(fitted) & (np.arange(len(train)) >= PERIOD)
                fit_metrics = metrics(train.to_numpy()[valid], fitted[valid], train)
                residual = train.to_numpy()[valid] - fitted[valid]
                lag = min(6, max(1, len(residual) // 5))
                lb = np.nan
                if np.std(residual) > 1e-12:
                    lb = acorr_ljungbox(residual, lags=[lag])["lb_pvalue"].iloc[0]
                fit_rows.append(dict(series=series, method=label, model=name,
                                     **fit_metrics, residual_lb_p=lb,
                                     elapsed_seconds=time.perf_counter() - started))
                for date, actual, pred in zip(test.index, test, forecast):
                    predictions.append(dict(series=series, method=label, date=date,
                                            actual=actual, forecast=pred))
                if label in ("E_best", "R_best", "Adaptive", "SeasonalNaive"):
                    _, values = fit_predict(y, args.future, lookup[name])
                    future[label] = values
                    for date, value in zip(future_index, values):
                        future_rows.append(dict(series=series, method=label, model=name,
                                                date=date, forecast=value))
            except Exception as error:
                log_issue(f"{series}:evaluation", name, error)
        full = [row for row in kpis if row["series"] == series and row["horizon"] == len(test)]
        test_scores = {row["method"]: row["MAE"] for row in full}
        result = dict(series=series, selected=adaptive, cv_best=cv_best,
                      reason=reason, test_MAE=test_scores.get("Adaptive", np.nan))
        for family in ("E", "R"):
            base, best = test_scores.get(f"{family}_base"), test_scores.get(f"{family}_best")
            result[f"{family}_improvement_pct"] = (
                100 * (base - best) / base if base is not None and best is not None and base > 0
                else np.nan)
        summary.append(result)
        print(f"Adaptive: {adaptive}; test MAE={result['test_MAE']:.3f}")
        print(reason)
        if series == "TOTAL":
            draw(series, y, train, forecasts, fits, future, cv, out)
    tables = dict(cv=pd.concat(all_cv, ignore_index=True),
                  properties=pd.DataFrame(properties_rows), summary=pd.DataFrame(summary),
                  test_metrics=pd.DataFrame(kpis), train_fit=pd.DataFrame(fit_rows),
                  test_predictions=pd.DataFrame(predictions), future=pd.DataFrame(future_rows),
                  issues=pd.DataFrame(ISSUES, columns=["stage", "model", "message"]))
    df.to_csv(out / "monthly_data.csv", encoding="utf-8-sig")
    df.iloc[:-args.test].corr().to_csv(out / "train_correlations.csv", encoding="utf-8-sig")
    with pd.ExcelWriter(out / "lab6_results.xlsx") as writer:
        for name, table in tables.items():
            table.to_csv(out / f"{name}.csv", index=False, encoding="utf-8-sig")
            table.to_excel(writer, sheet_name=name, index=False)
    notes = [
        f"Місячних спостережень: {len(df)}; test: {args.test}; future: {args.future}.",
        f"Нульове заповнення відсутніх місяць/регіон: {missing}.",
        "Потрібно підтвердити повноту журналу продажів, особливо крайніх місяців.",
        "Оптимізація і властивості ряду використовують лише train.",
        "E: SES/Holt/Holt-Winters; R: Ridge/Huber із трендом і Fourier-сезонністю.",
        "Adaptive: кандидати в межах 5% найкращого CV-MAE; властивості задають",
        "перевагу сімейства, а серед допустимих моделей обирається простіша.",
        "Пороги 0.5/0.6/5% — евристичні, не налаштовані на test.",
        "Порівняння Adaptive з CV_best перевіряє користь евристики на holdout.",
        "Усі моделі оцінюються на тих самих датах; відмови записано в issues.",
        "Train-fit KPI є внутрішніми: регресійна апроксимація й ETS fitted values",
        "мають різну інтерпретацію; основне порівняння — CV та незалежний test.",
        "Ljung-Box тут діагностичний, без поправки на оцінені параметри.",
        "p>0.05 не доводить білого шуму або адекватності моделі.",
        "Майбутні значення не мають KPI без фактичних спостережень.",
        "Параметри для future зафіксовано на train; моделі перевчено на всій історії.",
        "Від'ємне improvement_pct означає погіршення після оптимізації.",
        "Підсумки TOTAL та регіонів прогнозуються незалежно, без reconciliation.",
        "Негативні прогнози не обрізаються; це потрібно врахувати в інтерпретації.",
        "Програмні self-tests пройдено; це не доказ точності на нових даних.",
    ]
    (out / "methodology.txt").write_text("\n".join(notes), encoding="utf-8")
    print("\n", tables["summary"].to_string(index=False))
    print(f"\nРезультати: {out.resolve()}")


if __name__ == "__main__":
    main()