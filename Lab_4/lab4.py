"""Лабораторна №4: пошук закономірностей у Time Series (Data_Set_11, місячні продажі по регіонах).
Частина 1 - аналіз реальних даних; частина 2 - синтетичні ряди та перевірка адекватності."""
import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller, acf
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
from scipy.stats import ks_2samp
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 20)

# ===================== ЧАСТИНА 1: РЕАЛЬНІ ДАНІ =====================
def load(path="Data_Set_11.xlsx"):
    d = pd.read_excel(path, sheet_name="Orders")
    return d.pivot_table(index=pd.Grouper(key="Order Date", freq="MS"),
                         columns="Region", values="Sales", aggfunc="sum").fillna(0)

def props(df):
    """Показники: тренд/сезонність (сила за STL), ADF, ACF(1), CV, асиметрія."""
    r = {}
    for c in df:
        s = STL(df[c], period=12, robust=True).fit()
        sa, tr = 1 - s.resid.var() / (s.resid + s.seasonal).var(), 1 - s.resid.var() / (s.resid + s.trend).var()
        r[c] = dict(mean=df[c].mean(), cv=df[c].std() / df[c].mean(), skew=df[c].skew(),
                    acf1=acf(df[c], nlags=1)[1], adf_p=adfuller(df[c], regression="c")[1],
                    trend_str=max(tr, 0), season_str=max(sa, 0))
    return pd.DataFrame(r).T

def cluster(df, k=3):
    """Ієрархічна кластеризація за відстанню 1-corr."""
    D = 1 - df.corr().values; np.fill_diagonal(D, 0)
    Z = linkage(squareform(D, checks=False), "average")
    return Z, pd.Series(fcluster(Z, k, "maxclust"), index=df.columns)

def plots(df, Z, out="real_"):
    tot = df.sum(axis=1); s = STL(tot, period=12, robust=True).fit()
    s.plot().suptitle("STL-декомпозиція: сумарні продажі"); plt.savefig(out + "stl.png", dpi=120)
    ax = df.plot(figsize=(11, 5), title="Місячні продажі за регіонами"); ax.legend(fontsize=7); plt.savefig(out + "series.png", dpi=120)
    fig, a = plt.subplots(1, 2, figsize=(13, 5)); c = df.corr()
    im = a[0].imshow(c, cmap="coolwarm", vmin=-1, vmax=1); a[0].set_xticks(range(len(c)), c.columns, rotation=90); a[0].set_yticks(range(len(c)), c.columns)
    fig.colorbar(im, ax=a[0]); a[0].set_title("Кореляція"); dendrogram(Z, labels=list(df.columns), ax=a[1]); a[1].set_title("Дендрограма")
    plt.tight_layout(); plt.savefig(out + "corr_cluster.png", dpi=120)


# ===================== ЧАСТИНА 2: СИНТЕЗ І ПОРІВНЯННЯ =====================
def fit(df):
    t = np.arange(len(df)); m = {}; w = props(df)["season_str"] ** .5  # зсув сезонності до реальної сили (4 роки → шум у місячних середніх)
    for c in df:
        a, b = np.polyfit(t, df[c], 1); det = df[c] - (a * t + b)
        seas = det.groupby(df.index.month).mean() * w[c]; e = det - seas.reindex(df.index.month).values
        m[c] = (a, b, seas.values, np.corrcoef(e[:-1], e[1:])[0, 1], e.std())
    E = pd.DataFrame({c: df[c] - (m[c][0] * t + m[c][1]) - m[c][2][df.index.month - 1] for c in df})
    return m, np.linalg.cholesky(np.corrcoef(E.T) + 1e-6 * np.eye(len(m)))

def generate(m, L, idx, seed=0):
    rng = np.random.default_rng(seed); n, k = len(idx), len(m)
    z = rng.standard_normal((n, k)) @ L.T; out = {}
    for j, (c, (a, b, s, phi, sd)) in enumerate(m.items()):
        e = np.zeros(n); e[0] = z[0, j]
        for i in range(1, n): e[i] = phi * e[i - 1] + np.sqrt(1 - phi**2) * z[i, j]
        out[c] = a * np.arange(n) + b + s[idx.month - 1] + sd * e
    return pd.DataFrame(out, index=idx).clip(lower=0)


if __name__ == "__main__":
    real = load(); Z, cl = cluster(real); pr = props(real)
    print(pr.round(3)); print("\nКластери:\n", cl.sort_values()); print("\nКореляції:\n", real.corr().round(2))
    plots(real, Z)
    m, L = fit(real); syn = generate(m, L, real.index); ps = props(syn)
    cmp = pd.DataFrame({"mean_real": pr["mean"], "mean_syn": ps["mean"], "cv_real": pr["cv"], "cv_syn": ps["cv"],
                        "seas_real": pr["season_str"], "seas_syn": ps["season_str"], "acf1_real": pr["acf1"], "acf1_syn": ps["acf1"],
                        "KS_p": [ks_2samp(real[c], syn[c]).pvalue for c in real]})
    print("\n", cmp.round(3)); print("\nСер. |Δcorr| реальні vs синтетичні:", np.abs(real.corr() - syn.corr()).values.mean().round(3))
    print("MAPE середніх, %:", (100 * (cmp.mean_syn / cmp.mean_real - 1).abs().mean()).round(2))
    fig, a = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    real.sum(axis=1).plot(ax=a[0], title="Реальні (сума)"); syn.sum(axis=1).plot(ax=a[1], title="Синтетичні (сума)")
    plt.tight_layout(); plt.savefig("synth_vs_real.png", dpi=120)