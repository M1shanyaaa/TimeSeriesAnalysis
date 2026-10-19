"""ЛР3: AB, ABG та R&D — робастний банк із адаптивною пам'яттю."""
from pathlib import Path
from math import factorial
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("results_lab3")
OUT.mkdir(exist_ok=True)
START, CAL = 24, 48
RATIOS = (.001, .03, 1., 10.)
METHODS = ("AB", "ABG", "Robust_ABG", "Bank_fixed", "Bank_adaptive")


def variance(y):
    """Робастна оцінка R; наближення для локально лінійного сигналу."""
    z = np.diff(np.asarray(y), n=2)
    mad = 1.4826 * np.median(np.abs(z - np.median(z)))
    return max(float(mad**2 / 6), 1e-6)


def metrics(y, estimate):
    e = np.asarray(estimate) - np.asarray(y)
    return dict(RMSE=np.sqrt(np.mean(e**2)),
                MAE=np.mean(np.abs(e)), bias=np.mean(e))


def normalize(logw):
    w = np.exp(logw - np.max(logw))
    return w / w.sum()


class KF:
    """Звичайний або робастний KF. Один крок часу = один вимір."""

    def __init__(self, initial, d, R, q, robust=False):
        self.d, self.R, self.robust = d, R, robust
        self.F = np.array([
            [1 / factorial(j-i) if j >= i else 0.
             for j in range(d)] for i in range(d)
        ])
        g = np.array([1 / factorial(d-i) for i in range(d)])
        self.Q = q * np.outer(g, g)

        t = np.arange(len(initial), dtype=float) - len(initial) + 1
        X = np.column_stack([t**j / factorial(j) for j in range(d)])
        A = np.linalg.pinv(X)
        self.x = A @ initial
        self.P = R * A @ A.T

    def step(self, measurement):
        xp = self.F @ self.x
        Pp = self.F @ self.P @ self.F.T + self.Q
        e = measurement - xp[0]
        S = max(float(Pp[0, 0] + self.R), 1e-12)
        z2 = e*e / S

        trust = min(1., 5 / (4 + z2)) if self.robust else 1.
        Re = self.R / max(trust, 1e-12)
        K = Pp[:, 0] / (Pp[0, 0] + Re)
        self.x = xp + K*e

        B = np.eye(self.d)
        B[:, 0] -= K
        self.P = B @ Pp @ B.T + Re*np.outer(K, K)
        self.P = (self.P + self.P.T) / 2

        # Робастний predictive score для порівняння гіпотез.
        score = -.5*np.log(S) - 2.5*np.log1p(z2/4)
        return xp[0], self.x[0], S, score, trust


def run(y, method, R, ratio=.03):
    """
    Причинний банк гіпотез.
    prediction обчислюється з вагами ДО поточного виміру;
    filtered — після його отримання.
    """
    y = np.asarray(y, float)
    if len(y) <= START:
        raise ValueError("Недостатньо даних після стартового пакета.")

    bank = method.startswith("Bank")
    if bank:
        # Однакова сітка Q для обох банків.
        # Adaptive додатково перевіряє різні рівні шуму вимірювання.
        r_scales = (.25, 1., 4., 16.) if method == "Bank_adaptive" else (1.,)
        specs = [(d, q, rs)
                 for d in (1, 2, 3)
                 for q in RATIOS
                 for rs in r_scales]
    else:
        specs = [(2 if method == "AB" else 3, ratio, 1.)]

    robust = bank or method == "Robust_ABG"
    filters = [
        KF(y[:START], d, R*rs, R*q, robust=robust)
        for d, q, rs in specs
    ]
    weights = np.ones(len(filters)) / len(filters)
    rows = []

    for k in range(START, len(y)):
        # Однакове невелике відновлення ваг для ОБОХ банків.
        # Це фіксований параметр алгоритму, не R&D-адаптація.
        share = .01 if bank else 0.
        prior = (1-share)*weights + share/len(weights)

        prediction_values = np.array([f.step(y[k]) for f in filters])
        predictions, estimates, S, score, trust = prediction_values.T

        prediction = float(prior @ predictions)
        weights = normalize(np.log(np.maximum(prior, 1e-300)) + score)
        filtered = float(weights @ estimates)

        order_weights = [
            sum(weights[j] for j, (d, _, _) in enumerate(specs)
                if d == order)
            for order in (1, 2, 3)
        ]

        # Це зважені характеристики банку, а не ковaріації одного KF.
        effective_R = sum(
            weights[j]*R*rs for j, (_, _, rs) in enumerate(specs)
        )
        effective_Q = sum(
            weights[j]*R*q for j, (_, q, _) in enumerate(specs)
        )

        rows.append([
            k, prediction, filtered, share, float(prior @ trust),
            float(effective_R), float(effective_Q), *order_weights
        ])

    return pd.DataFrame(rows, columns=[
        "k", "prediction", "filtered", "memory_share", "trust",
        "effective_R", "effective_Q",
        "weight_level", "weight_AB", "weight_ABG"
    ])


def tune(y, R):
    """AB/ABG: стартовий Q/R обираємо лише на калібрувальних даних."""
    settings, rows = {}, []
    for method in ("AB", "ABG", "Robust_ABG"):
        for ratio in RATIOS:
            r = run(y[:CAL], method, R, ratio)
            score = metrics(y[START:CAL], r.prediction)["RMSE"]
            rows.append(dict(method=method, ratio=ratio, RMSE=score))
        best = min((r for r in rows if r["method"] == method),
                   key=lambda r: r["RMSE"])
        settings[method] = best["ratio"]
    return settings, pd.DataFrame(rows)


def synthetic(seed, scenario, n=240):
    """Еталон відомий тільки оцінювачу, але не фільтрам."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    truth = (
        np.full(n, 100.) if scenario == "constant"
        else 100 + .015*t + .00015*t*t
    )
    if scenario == "change":
        truth += 2*(t >= 100) - .04*np.maximum(t-160, 0)

    sigma = rng.uniform(.15, .6)
    noise = np.full(n, sigma)
    if scenario == "noise":
        noise[t >= 140] *= 3
    y = truth + rng.normal(size=n)*noise

    if scenario in ("change", "noise"):
        idx = rng.choice(np.arange(CAL, n), 12, replace=False)
        y[idx] += rng.choice([-1., 1.], len(idx))*rng.uniform(
            5*sigma, 10*sigma, len(idx)
        )
    return truth, y


def plot_case(x, y, results, filename, truth=None):
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    x = np.asarray(x)
    axes[0].plot(x, y, color="gray", alpha=.5, label="Виміри")
    if truth is not None:
        axes[0].plot(x, truth, "k", lw=2, label="Істинний сигнал")

    for name, r in results.items():
        axes[0].plot(x[r.k.to_numpy(int)], r.filtered, label=name)

    r = results["Bank_adaptive"]
    xx = x[r.k.to_numpy(int)]
    for col in ("weight_level", "weight_AB", "weight_ABG"):
        axes[1].plot(xx, r[col], label=col)

    axes[2].plot(xx, r.effective_R, label="Зважене R")
    axes[2].plot(xx, r.effective_Q, label="Зважене Q")
    axes[2].set_yscale("log")

    for ax in axes:
        ax.grid(alpha=.25)
        ax.legend(ncol=2)
    axes[0].set_ylabel("Значення")
    axes[1].set_ylabel("Вага моделі")
    axes[2].set_ylabel("Параметр шуму")
    fig.tight_layout()
    fig.savefig(OUT / filename, dpi=170)
    plt.close(fig)

def check_unbiasedness(repeats=200):
    """
    Відомі R, фіксоване Q, нульове середнє шуму.
    AB перевіряється на лінійному, ABG — на квадратичному сигналі.
    """
    rows = []
    t = np.arange(120, dtype=float)
    sigma = .3

    for method in ("AB", "ABG"):
        truth = 100 + .02*t
        if method == "ABG":
            truth += .0002*t*t

        biases = []
        for seed in range(5000, 5000+repeats):
            rng = np.random.default_rng(seed)
            y = truth + rng.normal(0, sigma, len(t))
            r = run(y, method, R=sigma**2, ratio=.03)
            biases.append(np.mean(r.filtered.to_numpy()-truth[START:]))

        biases = np.asarray(biases)
        half = 1.96*biases.std(ddof=1)/np.sqrt(repeats)
        rows.append({
            "method": method,
            "mean_bias": biases.mean(),
            "low95": biases.mean()-half,
            "high95": biases.mean()+half,
        })

    table = pd.DataFrame(rows)
    table.to_csv(OUT / "classical_unbiasedness.csv", index=False)
    print("\nКЛАСИЧНІ ФІЛЬТРИ: ПЕРЕВІРКА ЗМІЩЕННЯ\n", table.round(6))


def verify(repeats=30):
    records = []
    for scenario in ("constant", "quadratic", "change", "noise"):
        for seed in range(1000, 1000+repeats):
            truth, y = synthetic(seed, scenario)
            records.append({
                "scenario": scenario, "seed": seed, "method": "Measurements",
                **metrics(truth[CAL:], y[CAL:])
            })
            R = variance(y[:START])
            settings, _ = tune(y, R)

            for method in METHODS:
                r = run(y, method, R, settings.get(method, .03))
                test = r[r.k >= CAL]
                records.append({
                    "scenario": scenario, "seed": seed, "method": method,
                    **metrics(truth[CAL:], test.filtered)
                })

    records = pd.DataFrame(records)
    paired = records.pivot(
        index=["scenario", "seed"], columns="method", values="RMSE"
    )
    delta = paired["Bank_adaptive"] - paired["Bank_fixed"]
    comparison = delta.groupby(level="scenario").agg(["mean", "std", "count"])
    half = 1.96 * comparison["std"] / np.sqrt(comparison["count"])
    comparison["low95"] = comparison["mean"] - half
    comparison["high95"] = comparison["mean"] + half
    comparison.to_csv(OUT / "paired_bank_comparison.csv")
    print("\nΔRMSE = Adaptive − Fixed; від'ємне — краще:\n",
          comparison.round(5))
    records.to_csv(OUT / "synthetic_runs.csv", index=False)

    grouped = records.groupby(["scenario", "method"])
    summary = grouped[["RMSE", "MAE", "bias"]].mean()
    se = grouped.bias.std() / np.sqrt(repeats)
    summary["bias_low95"] = summary.bias - 1.96*se
    summary["bias_high95"] = summary.bias + 1.96*se
    summary.to_csv(OUT / "synthetic_summary.csv")

    # Перевірка причинності та симетрії адаптивного алгоритму.
    truth, y = synthetic(2000, "change")

    R = variance(y[:START])
    full = run(y, "Bank_adaptive", R)
    prefix = run(y[:120], "Bank_adaptive", R)
    np.testing.assert_allclose(prefix.filtered, full.filtered[:len(prefix)])
    reflected = run(200-y, "Bank_adaptive", R)
    np.testing.assert_allclose(reflected.filtered, 200-full.filtered, atol=1e-7)
    assert np.isfinite(full.filtered).all()

    # Безшумний квадратичний сигнал точно відтворюється ABG.
    t = np.arange(80)
    exact = 100 + .02*t + .001*t*t
    r = run(exact, "ABG", .09)
    np.testing.assert_allclose(r.filtered, exact[START:], atol=1e-8)

    results = {m: run(y, m, R) for m in METHODS}
    plot_case(np.arange(len(y)), y, results, "synthetic.png", truth)
    print("\nСИНТЕТИКА: помилка ФІЛЬТРАЦІЇ\n", summary.round(4))


def main():
    df = pd.read_csv("data/parsed_data.csv", parse_dates=["date"])
    df = df[["date", "cpi"]].sort_values("date").reset_index(drop=True)
    y = df.cpi.to_numpy(float)
    if len(y) < 72 or not np.isfinite(y).all() or (y <= 0).any():
        raise ValueError("Потрібно >=72 коректних місячних значень.")
    expected = pd.date_range(df.date.iloc[0], periods=len(df), freq="MS")
    if not np.array_equal(expected.to_numpy(), df.date.to_numpy()):
        raise ValueError("Дати мають бути щомісячними, без дублікатів і пропусків.")

    R = variance(y[:START])
    settings, calibration = tune(y, R)
    calibration.to_csv(OUT / "calibration.csv", index=False)
    pd.DataFrame([
        {"method": m, "R": R, "Q_R": q} for m, q in settings.items()
    ]).to_csv(OUT / "settings.csv", index=False)

    results, rows = {}, []
    for method in METHODS:
        r = run(y, method, R, settings.get(method, .03))
        r["date"] = df.date.iloc[r.k.to_numpy(int)].to_numpy()
        r.to_csv(OUT / f"{method}.csv", index=False)
        test = r[r.k >= CAL]
        rows.append({"method": method,
                     **metrics(y[CAL:], test.prediction)})
        results[method] = r

    rows.append({"method": "naive_last",
                 **metrics(y[CAL:], y[CAL-1:-1])})
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "real_prediction_metrics.csv", index=False)
    plot_case(df.date, y, results, "real_filters.png")
    print("\nІСЦ: помилка однокрокового ПРОГНОЗУ\n", table.round(4))
    verify()
    check_unbiasedness()
    print("\nТести причинності, симетрії та точного тренду пройдено.")
    print("Результати:", OUT.resolve())


if __name__ == "__main__":
    main()