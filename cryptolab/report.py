"""Generate a self-contained HTML research report (charts embedded as PNG)."""

from __future__ import annotations

import base64
import datetime as dt
import html
import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

from . import __version__, benchmark, risk, stats  # noqa: E402
from . import indicators as ind  # noqa: E402
from . import signals as sig  # noqa: E402
from . import strategies as st  # noqa: E402
from .config import FEE_BPS, MARKET_PROXY, SLIPPAGE_BPS  # noqa: E402
from .data import validate_ohlcv  # noqa: E402
from .portfolio import ledoit_wolf, risk_contributions, risk_parity  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # fixed categorical order
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
SEQ = LinearSegmentedColormap.from_list(
    "seq_blue", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])
BULL_FILL, BEAR_FILL = "#1baf7a", "#e34948"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 9,
    "font.family": ["DejaVu Sans"], "lines.linewidth": 1.6, "legend.frameon": False,
})


def _png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _img(b64: str, alt: str) -> str:
    return f'<img alt="{html.escape(alt)}" src="data:image/png;base64,{b64}">'


def _label_line_ends(ax, series: dict[str, pd.Series], colors):
    ends = sorted(((s.dropna().iloc[-1], name, c) for (name, s), c in zip(series.items(), colors)),
                  key=lambda x: x[0])
    ylim = ax.get_ylim()
    log = ax.get_yscale() == "log"
    to_ax = (lambda v: (np.log(v) - np.log(ylim[0])) / (np.log(ylim[1]) - np.log(ylim[0]))) if log \
        else (lambda v: (v - ylim[0]) / (ylim[1] - ylim[0]))
    last = -1.0
    for v, name, _ in ends:
        y = max(to_ax(v), last + 0.06)
        last = y
        ax.annotate(name, xy=(1.01, y), xycoords="axes fraction", color=INK2, fontsize=8, va="center")


def chart_equity(results: dict[str, pd.Series], title: str) -> str:
    fig, ax = plt.subplots(figsize=(9, 3.6))
    eq = {k: (1 + r).cumprod() for k, r in results.items()}
    for (name, e), c in zip(eq.items(), SERIES):
        ax.plot(e.index, e, color=c, label=name)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}×"))
    ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: f"{v:g}×" if f"{v:.0e}"[0] in "25" else ""))
    ax.set_title(title)
    ax.set_ylabel("Growth of 1 (log scale)")
    ax.legend(loc="upper left", fontsize=8)
    _label_line_ends(ax, eq, SERIES)
    return _png(fig)


def chart_drawdowns(results: dict[str, pd.Series]) -> str:
    fig, ax = plt.subplots(figsize=(9, 2.6))
    for (name, r), c in zip(results.items(), SERIES):
        ax.plot(r.index, risk.drawdown(r) * 100, color=c, label=name, linewidth=1.2)
    ax.set_title("Drawdown from previous peak")
    ax.set_ylabel("%")
    ax.legend(loc="lower left", fontsize=8, ncol=len(results))
    return _png(fig)


def chart_heatmap(mat: pd.DataFrame, title: str, fmt: str = "{:.2f}", vmin=None, vmax=None,
                  xlabel: str = "", ylabel: str = "") -> str:
    fig, ax = plt.subplots(figsize=(0.48 * mat.shape[1] + 2.2, 0.42 * mat.shape[0] + 1.2))
    ax.grid(False)
    im = ax.imshow(mat.to_numpy(float), cmap=SEQ, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(mat.shape[1]), [str(c) for c in mat.columns], rotation=0)
    ax.set_yticks(range(mat.shape[0]), [str(i) for i in mat.index])
    lo, hi = im.get_clim()
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat.iat[i, j]
            if pd.notna(v):
                dark = (v - lo) / (hi - lo + 1e-12) > 0.55
                ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=7,
                        color="#ffffff" if dark else INK)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02).outline.set_visible(False)
    return _png(fig)


def chart_regime(close: pd.Series, p_bull: pd.Series) -> str:
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(9, 4.2), sharex=True,
                                  gridspec_kw={"height_ratios": [2.2, 1]})
    c = close.loc[p_bull.index[0]:]
    ax.plot(c.index, c, color=INK, linewidth=1.1)
    ax.set_yscale("log")
    bull = (p_bull > 0.5).reindex(c.index).fillna(False)
    ax.fill_between(c.index, 0, 1, where=bull, transform=ax.get_xaxis_transform(),
                    color=BULL_FILL, alpha=0.12, linewidth=0, label="Bull regime (p > 0.5)")
    ax.fill_between(c.index, 0, 1, where=~bull, transform=ax.get_xaxis_transform(),
                    color=BEAR_FILL, alpha=0.10, linewidth=0, label="Bear regime")
    ax.set_title(f"{MARKET_PROXY} with out-of-sample regime (walk-forward HMM, causal filter)")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}" if f"{v:.0e}"[0] in "25" else ""))
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=8)
    ax2.plot(p_bull.index, p_bull, color=SERIES[0], linewidth=1.0)
    ax2.axhline(0.5, color=AXIS, linewidth=0.8)
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("P(bull)")
    return _png(fig)


def chart_indicators(df: pd.DataFrame, sym: str, days: int = 365) -> str:
    d = df.iloc[-days - 250:]
    c = d["close"]
    bb = ind.bollinger(c)
    s50 = ind.sma(c, 50)
    r = ind.rsi(c)
    m = ind.macd(c)
    sl = slice(-days, None)
    fig, axes = plt.subplots(3, 1, figsize=(9, 5.6), sharex=True,
                             gridspec_kw={"height_ratios": [2.4, 1, 1]})
    ax = axes[0]
    ax.fill_between(c.index[sl], bb["lower"].iloc[sl], bb["upper"].iloc[sl], color=SERIES[0],
                    alpha=0.10, linewidth=0, label="Bollinger (20, 2)")
    ax.plot(c.index[sl], c.iloc[sl], color=INK, linewidth=1.2, label="Close")
    ax.plot(c.index[sl], s50.iloc[sl], color=SERIES[1], linewidth=1.2, label="SMA 50")
    ax.set_title(f"{sym}: indicator panel, last {days} days")
    ax.legend(loc="upper left", fontsize=8, ncol=3)
    axes[1].plot(r.index[sl], r.iloc[sl], color=SERIES[0], linewidth=1.1)
    for lvl in (30, 70):
        axes[1].axhline(lvl, color=AXIS, linewidth=0.8, linestyle="--")
    axes[1].set_ylim(0, 100)
    axes[1].set_ylabel("RSI 14")
    h = m["hist"].iloc[sl]
    axes[2].bar(h.index, h, color=np.where(h >= 0, SERIES[2], "#e34948"), width=1.0)
    axes[2].plot(m.index[sl], m["macd"].iloc[sl], color=INK, linewidth=1.0, label="MACD")
    axes[2].plot(m.index[sl], m["signal"].iloc[sl], color=SERIES[1], linewidth=1.0, label="Signal")
    axes[2].set_ylabel("MACD")
    axes[2].legend(loc="upper left", bbox_to_anchor=(0, 1.32), fontsize=8, ncol=2)
    return _png(fig)


def chart_risk_contrib(rc: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(9, 2.8))
    x = np.arange(len(rc))
    w = 0.38
    for i, (col, c) in enumerate(zip(rc.columns, SERIES)):
        ax.bar(x + (i - 0.5) * w, rc[col] * 100, width=w - 0.04, color=c, label=col)
    ax.set_xticks(x, rc.index)
    ax.set_ylabel("% of portfolio risk")
    ax.set_title("Risk contribution by asset: equal capital vs. equal risk")
    ax.legend(fontsize=8)
    return _png(fig)


def _fmt_table(df: pd.DataFrame, pct: list[str] = (), dec2: list[str] = (), dec0: list[str] = ()) -> str:
    out = df.copy().astype(object)
    for c in df.columns:
        col = df[c]
        if c in pct:
            out[c] = [f"{v:.1%}" if pd.notna(v) else "–" for v in col]
        elif c in dec2:
            out[c] = [f"{v:,.2f}" if pd.notna(v) else "–" for v in col]
        elif c in dec0:
            out[c] = [f"{v:,.0f}" if pd.notna(v) else "–" for v in col]
    return out.to_html(classes="tbl", border=0, escape=True)


def build_report(universe: dict[str, pd.DataFrame], source_label: str, out_path: Path,
                 synthetic: bool, train_days: int = 730, test_days: int = 90,
                 run_benchmark: bool = True) -> dict:
    clean, quality = {}, []  # cleaned data plus a log of what was fixed
    for s, d in universe.items():
        c, rep = validate_ohlcv(d, s)
        clean[s] = c.dropna(subset=["close"])
        quality.append(rep)
    m = st.Market(clean)
    idx = m.prices.index
    eval_start = idx[train_days]

    # strategies (everything is scored from eval_start, after the first training window)
    live = m.prices.notna()
    ew_w = live.div(live.sum(axis=1), axis=0)
    res = {
        "BTC buy & hold": st.btc_hold(m),
        "Equal-weight": st.equal_weight(m),
        "Composite trend": st.composite_trend(m),
        "Risk parity": st.risk_parity_strategy(m),
    }
    regime_res, p_bull, regime_folds = st.regime_overlay(m, ew_w, train_days, test_days)
    wf_res, wf_folds = st.xs_momentum_walk_forward(m, train_days, test_days)
    res["Regime-gated EW (WF)"] = regime_res
    res["XS momentum (WF)"] = wf_res

    rets = {k: v.returns.loc[eval_start:] for k, v in res.items()}
    summary = pd.DataFrame({k: risk.performance_summary(r.rename(k)).iloc[0] for k, r in rets.items()}).T
    for k, v in res.items():
        summary.loc[k, "Turnover/yr"] = v.turnover.loc[eval_start:].mean() * 365
        summary.loc[k, "Cost drag/yr"] = v.costs.loc[eval_start:].mean() * 365
        summary.loc[k, "Avg exposure"] = v.weights.loc[eval_start:].sum(axis=1).mean()

    # how much do costs matter, and how did each year go?
    base_bps = FEE_BPS + SLIPPAGE_BPS
    cost_levels = [0.0, base_bps, 2 * base_bps, 4 * base_bps]
    cost_tbl = pd.DataFrame({
        f"{bps:.0f} bps" + (" (base)" if bps == base_bps else ""):
            {k: risk.sharpe((v if bps == base_bps else v.with_costs(bps)).returns.loc[eval_start:])
             for k, v in res.items()}
        for bps in cost_levels})
    years_tbl = risk.calendar_returns(pd.DataFrame(rets))
    years_tbl.index.name = None

    # is any of it real? (bootstrap CIs, PSR, deflated Sharpe over the whole grid)
    sweep = st.parameter_sweep(m)
    best = sweep.loc[sweep["sharpe"].idxmax()]
    best_ret = st.xs_momentum(m, int(best["lookback"]), int(best["top_k"])).returns
    warmup = pd.Timedelta(days=int(best["lookback"]) + 60)
    best_ret = best_ret[best_ret.index >= idx[0] + warmup]  # same sample the sweep used
    dsr = stats.deflated_sharpe(best_ret, sweep["sharpe_pp"].to_numpy())
    ew_full = res["Equal-weight"].returns.loc[best_ret.index[0]:]
    ew_full_sr = risk.sharpe(ew_full)
    sig_rows = {}
    for k, r in rets.items():
        ci = stats.stationary_bootstrap_sharpe(r, n_boot=1000)
        sig_rows[k] = {"Sharpe": risk.sharpe(r), "95% CI low": ci["low"], "95% CI high": ci["high"],
                       "PSR (SR>0)": stats.probabilistic_sharpe(r),
                       "Min track record (yrs)": stats.min_track_record(r) / 365}
    sig_tbl = pd.DataFrame(sig_rows).T

    # regimes, as fitted in the most recent fold
    last_fold = regime_folds[-1].info
    reg_tbl = pd.DataFrame({
        "Mean 10d log return (daily)": [last_fold["bear_mean_ret"], last_fold["bull_mean_ret"]],
        "Daily vol": [last_fold["bear_vol"], last_fold["bull_vol"]],
        "Expected duration (days)": last_fold["durations"],
    }, index=["Bear / stress state", "Bull / calm state"])

    # risk and correlation
    r_all = m.returns.loc[eval_start:]
    corr = r_all.corr()
    stress = risk.downside_correlation(r_all.dropna(how="all"), MARKET_PROXY)
    calm_corr = r_all.corr()[MARKET_PROXY].drop(MARKET_PROXY)
    beta = pd.Series({s: risk.rolling_beta(r_all[s], r_all[MARKET_PROXY], 180).iloc[-1]
                      for s in r_all.columns if s != MARKET_PROXY})
    asset_risk = risk.performance_summary(r_all)[["CAGR", "Vol", "Sharpe", "MaxDD", "VaR95", "CVaR95",
                                                   "CF-VaR95", "Skew", "Kurt"]]
    asset_risk["Corr w/ BTC"] = calm_corr
    asset_risk["Corr w/ BTC, stress days"] = stress
    asset_risk["Beta (180d)"] = beta

    win = m.returns.iloc[-90:].dropna(axis=1)
    cov, delta = ledoit_wolf(win)
    w_rp = risk_parity(cov)
    w_ew = pd.Series(1 / len(cov), index=cov.index)
    rc = pd.DataFrame({"Equal capital": risk_contributions(w_ew, cov),
                       "Risk parity": risk_contributions(w_rp, cov)}, index=cov.index)

    screen = sig.latest_screen(clean)  # today's snapshot
    bench = benchmark.run(clean, repeats=3) if run_benchmark else None

    # charts
    eq_a = {k: rets[k] for k in ["BTC buy & hold", "Equal-weight", "Regime-gated EW (WF)", "XS momentum (WF)"]}
    eq_b = {k: rets[k] for k in ["Equal-weight", "Composite trend", "Risk parity"]}
    charts = {
        "eq_a": chart_equity(eq_a, "Benchmarks vs. walk-forward strategies"),
        "dd_a": chart_drawdowns(eq_a),
        "eq_b": chart_equity(eq_b, "Volatility-managed strategies vs. equal-weight"),
        "regime": chart_regime(m.prices[MARKET_PROXY], p_bull),
        "sweep": chart_heatmap(sweep.pivot(index="lookback", columns="top_k", values="sharpe"),
                               "Sharpe across the momentum grid (full sample)", xlabel="Assets held (k)",
                               ylabel="Lookback (days)"),
        "corr": chart_heatmap(corr, "Return correlation matrix"),
        "rc": chart_risk_contrib(rc),
        "ind": chart_indicators(clean[MARKET_PROXY], MARKET_PROXY),
    }

    # text for the key findings
    best_name = summary["Sharpe"].idxmax()
    wf_params = pd.DataFrame([{"Test window": f"{f.test_start.date()} → {f.test_end.date()}",
                               "Lookback": f.info["lookback"], "Top k": f.info["top_k"],
                               "In-sample Sharpe": f.info["in_sample_sharpe"]} for f in wf_folds])
    stable = wf_params["Lookback"].nunique()
    now_bull = float(p_bull.iloc[-1])
    regime_word = "bull / calm" if now_bull > 0.5 else "bear / stressed"
    n_bull = int((screen["Composite"].astype(float) > 0.1).sum())

    banner = ("<div class='banner'><strong>Synthetic data.</strong> This report was generated from the "
              "built-in simulated market so the pipeline can run offline. Numbers illustrate the method "
              "and are not market results. Run <code>python -m cryptolab report --source binance</code> "
              "for live data.</div>") if synthetic else ""

    findings = [
        f"Highest risk-adjusted return over the evaluation window: <b>{html.escape(best_name)}</b> "
        f"(Sharpe {summary.loc[best_name, 'Sharpe']:.2f}, max drawdown {summary.loc[best_name, 'MaxDD']:.0%}).",
        f"Best of {len(sweep)} momentum configurations had Sharpe {best['sharpe']:.2f}. Deflated for the number "
        f"of trials (luck alone would be expected to reach ≈ {dsr['SR0_annual']:.2f}), the probability its true "
        f"Sharpe is above zero is <b>{dsr['DSR']:.0%}</b>. Equal-weight buy-and-hold alone scored "
        f"{ew_full_sr:.2f} on the same sample" + (", so much of that is market exposure rather than selection skill."
            if ew_full_sr >= 0.6 * best["sharpe"] else ", so selection adds clearly more than market exposure."),
        f"Walk-forward selection chose {stable} different lookbacks across {len(wf_folds)} folds; "
        f"out-of-sample Sharpe {summary.loc['XS momentum (WF)', 'Sharpe']:.2f} vs. "
        f"{best['sharpe']:.2f} for the hindsight-optimal setting.",
        f"Market regime now: <b>{regime_word}</b> (P(bull) = {now_bull:.0%}). "
        f"{n_bull} of {len(screen)} assets carry a positive composite signal.",
        f"Average correlation with BTC is {calm_corr.mean():.2f} on all days and {stress.mean():.2f} on BTC's "
        f"worst 10% of days"
        + (": diversification thins out when it is needed most." if stress.mean() > calm_corr.mean() + 0.02
           else "; with correlations this high, spreading capital across coins diversifies little."),
    ]
    cost_hit = (cost_tbl.iloc[:, 1] - cost_tbl.iloc[:, 2]).drop("BTC buy & hold", errors="ignore")
    most_sensitive = cost_hit.idxmax()
    sharpe_base, sharpe_double = cost_tbl.loc[most_sensitive].iloc[1], cost_tbl.loc[most_sensitive].iloc[2]
    findings.append(f"Doubling trading costs to {2 * base_bps:.0f} bps hurts <b>{html.escape(most_sensitive)}</b> most "
                    f"(Sharpe {sharpe_base:.2f} → {sharpe_double:.2f}), "
                    f"in line with its turnover of {summary.loc[most_sensitive, 'Turnover/yr']:.0f}× a year.")
    if bench:
        findings.append(f"Vectorised indicator engine computes the full indicator set for {bench['assets']} assets "
                        f"({bench['rows']:,} rows) in {bench['vectorised_s'] * 1000:.0f} ms, "
                        f"<b>{bench['speedup_x']:.1f}× faster</b> than a loop implementation "
                        f"({bench['time_saved_pct']:.0f}% less compute time).")

    q_tbl = pd.DataFrame([{"Asset": q.symbol, "First": q.first.date() if q.first is not None else None,
                           "Last": q.last.date() if q.last is not None else None,
                           "Duplicates": q.duplicates, "Missing days": q.missing_days,
                           "OHLC fixes": q.ohlc_violations, "Extreme moves": q.extreme_moves}
                          for q in quality]).set_index("Asset")

    pct_cols = ["CAGR", "Vol", "MaxDD", "VaR95", "CVaR95", "CF-VaR95", "Hit rate", "Turnover/yr",
                "Cost drag/yr", "Avg exposure"]
    perf_html = _fmt_table(summary.drop(columns=["Days"]), pct=pct_cols,
                           dec2=["Sharpe", "Sortino", "Calmar", "Skew", "Kurt"], dec0=["MaxDD days"])
    screen_html = _fmt_table(screen, pct=["1D %", "30D %", "vs 200D", "Vol 30D"],
                             dec2=["Price", "Hurst", "Momentum", "Trend", "Composite"], dec0=["RSI", "ADX"])
    sig_html = _fmt_table(sig_tbl, pct=["PSR (SR>0)"], dec2=["Sharpe", "95% CI low", "95% CI high",
                                                             "Min track record (yrs)"])
    risk_html = _fmt_table(asset_risk, pct=["CAGR", "Vol", "MaxDD", "VaR95", "CVaR95", "CF-VaR95"],
                           dec2=["Sharpe", "Skew", "Kurt", "Corr w/ BTC", "Corr w/ BTC, stress days",
                                 "Beta (180d)"])
    wf_html = _fmt_table(wf_params.set_index("Test window"), dec2=["In-sample Sharpe"])
    cost_html = _fmt_table(cost_tbl, dec2=list(cost_tbl.columns))
    years_html = _fmt_table(years_tbl, pct=list(years_tbl.columns))
    reg_html = _fmt_table(reg_tbl, pct=["Mean 10d log return (daily)", "Daily vol"], dec2=["Expected duration (days)"])
    q_html = q_tbl.to_html(classes="tbl", border=0)

    page = TEMPLATE.format(
        title="Crypto Market Analysis",
        subtitle=f"{len(clean)} assets · {idx[0].date()} → {idx[-1].date()} · evaluation window from "
                 f"{eval_start.date()} · source: {html.escape(source_label)}",
        banner=banner,
        findings="".join(f"<li>{f}</li>" for f in findings),
        screen=screen_html, perf=perf_html, costs=cost_html, years=years_html, sig=sig_html, wf=wf_html, reg=reg_html,
        risk=risk_html, quality=q_html,
        eq_a=_img(charts["eq_a"], "Equity curves of benchmarks and walk-forward strategies"),
        dd_a=_img(charts["dd_a"], "Drawdowns"),
        eq_b=_img(charts["eq_b"], "Equity curves of volatility-managed strategies"),
        regime=_img(charts["regime"], "BTC price with regime shading"),
        sweep=_img(charts["sweep"], "Sharpe heatmap across parameter grid"),
        corr=_img(charts["corr"], "Correlation heatmap"),
        rc=_img(charts["rc"], "Risk contributions"),
        ind=_img(charts["ind"], "Indicator panel"),
        shrink=f"{delta:.2f}", fee=FEE_BPS, slip=SLIPPAGE_BPS, train=train_days, test=test_days,
        n_trials=len(sweep), version=__version__,
        generated=dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(page, encoding="utf-8")
    return {"summary": summary, "dsr": dsr, "benchmark": bench, "screen": screen,
            "significance": sig_tbl, "costs": cost_tbl, "years": years_tbl, "path": out_path}


TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
:root {{ --surface:#fcfcfb; --page:#f9f9f7; --ink:#0b0b0b; --ink2:#52514e; --muted:#898781;
        --line:#e1e0d9; --accent:#2a78d6; --warn-bg:#fff4dc; --warn-line:#eda100; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--page); color:var(--ink);
       font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif; }}
main {{ max-width:1080px; margin:0 auto; padding:32px 16px 64px; }}
header h1 {{ font-size:28px; margin:0 0 4px; letter-spacing:-0.01em; }}
header p {{ color:var(--ink2); margin:0; }}
section {{ background:var(--surface); border:1px solid var(--line); border-radius:10px;
          padding:20px 22px; margin-top:20px; }}
h2 {{ font-size:18px; margin:0 0 4px; }}
.lede {{ color:var(--ink2); margin:0 0 14px; max-width:78ch; }}
img {{ max-width:100%; height:auto; display:block; margin:10px 0; }}
.banner {{ background:var(--warn-bg); border-left:4px solid var(--warn-line); padding:10px 14px;
          border-radius:6px; margin-top:18px; }}
ul.findings li {{ margin:6px 0; }}
.scroll {{ overflow-x:auto; }}
table.tbl {{ border-collapse:collapse; font-size:12.5px; font-variant-numeric:tabular-nums; width:100%; }}
table.tbl th, table.tbl td {{ padding:6px 8px; border-bottom:1px solid var(--line); text-align:right;
                              white-space:nowrap; }}
table.tbl th:first-child, table.tbl td:first-child {{ text-align:left; }}
table.tbl thead th {{ color:var(--ink2); font-weight:600; border-bottom:1px solid #c3c2b7; }}
code {{ background:#f0efec; padding:1px 5px; border-radius:4px; font-size:13px; }}
.grid2 {{ display:grid; grid-template-columns:1fr 1fr; gap:18px; }}
@media (max-width:760px) {{ .grid2 {{ grid-template-columns:1fr; }} }}
footer {{ color:var(--muted); font-size:12.5px; margin-top:28px; }}
</style></head>
<body><main>
<header><h1>{title}</h1><p>{subtitle}</p></header>
{banner}

<section><h2>Key findings</h2>
<p class="lede">Generated automatically from the analysis below.</p>
<ul class="findings">{findings}</ul></section>

<section><h2>1 · Market screen</h2>
<p class="lede">Latest state of every asset, ranked by composite signal (50% volatility-normalised time-series
momentum over 20/60/120 days, 35% EMA-20/100 trend gated by ADX, 15% mean reversion active only in non-trending
markets). Hurst &lt; 0.5 indicates mean-reverting behaviour over the last 180 days; &gt; 0.5 indicates persistence.</p>
<div class="scroll">{screen}</div></section>

<section><h2>2 · Strategy performance</h2>
<p class="lede">All strategies are long-only spot, pay {fee:.0f} bps fees + {slip:.0f} bps slippage on every trade,
and are measured over the same out-of-sample window. Signals formed at the close of day t earn day t+1's return.</p>
{eq_a}{dd_a}{eq_b}
<div class="scroll">{perf}</div>
<h3 style="font-size:15px">Year by year</h3>
<p class="lede">Compounded return in each calendar year of the evaluation window (the first and last years are
partial).</p>
<div class="scroll">{years}</div>
<h3 style="font-size:15px">How much do costs matter?</h3>
<p class="lede">Sharpe ratio of each strategy re-run with the same trades at different one-way cost assumptions
(fees plus slippage). A strategy whose Sharpe collapses between the base case and double costs is living off
turnover, not signal.</p>
<div class="scroll">{costs}</div></section>

<section><h2>3 · Is any of this real? Statistical controls</h2>
<p class="lede">A backtest Sharpe ratio is a noisy estimate, and the best of many tried configurations is biased
upwards. The table gives stationary-bootstrap 95% confidence intervals (blocks preserve volatility clustering),
the Probabilistic Sharpe Ratio (adjusted for skew and fat tails) and the track record needed for 95% confidence.
The heatmap shows all {n_trials} momentum configurations tested; the Deflated Sharpe Ratio in the key findings
accounts for every one of them.</p>
<div class="scroll">{sig}</div>
<div class="grid2"><div>{sweep}</div><div><p class="lede" style="margin-top:12px"><b>Walk-forward parameter
choice.</b> Every {test} days the lookback and number of holdings are re-selected using only the preceding data
(expanding window, first {train} days for training), then traded blind.</p><div class="scroll">{wf}</div></div></div>
</section>

<section><h2>4 · Market regimes</h2>
<p class="lede">A two-state Gaussian Hidden Markov Model (Baum-Welch EM, written from scratch in NumPy) on BTC's
10-day mean return and log 20-day volatility. It is refitted every {test} days on past data only, and the shading
uses the forward-filtered probability, which never sees the future. Parameters below are from the latest fit.</p>
{regime}<div class="scroll">{reg}</div></section>

<section><h2>5 · Risk and dependence</h2>
<p class="lede">Historical, Gaussian and Cornish-Fisher VaR at 95% (daily); CVaR is the average loss beyond VaR.
Excess kurtosis well above 0 means normal-distribution risk models understate tail losses.</p>
<div class="scroll">{risk}</div>
<div class="grid2"><div>{corr}</div><div>{rc}<p class="lede">Covariance over the last 90 days with Ledoit-Wolf
shrinkage intensity {shrink}. Equal capital concentrates risk in the most volatile names; equal risk contribution
spreads it.</p></div></div></section>

<section><h2>6 · Indicator panel</h2>{ind}</section>

<section><h2>7 · Methodology and limitations</h2>
<ul>
<li><b>No look-ahead.</b> Every indicator and signal is unit-tested by scrambling all data after a cut-off and
asserting that values before it do not change. The backtester applies day-t weights to day t+1 returns.</li>
<li><b>Costs and drift.</b> Turnover is measured against weights after they drift with prices, so costs are not
understated. Rebalancing is weekly with a 1% no-trade band.</li>
<li><b>Survivorship bias.</b> The universe is today's large caps. Coins that collapsed (e.g. LUNA, FTT) are absent,
which flatters any long-only crypto result.</li>
<li><b>Not modelled:</b> funding rates and shorting, exchange outages, tax, capacity and market impact beyond a flat
slippage charge.</li>
<li><b>Sample size.</b> Crypto has had only a few full market cycles; the wide Sharpe confidence intervals
reflect that.</li>
</ul>
<h3 style="font-size:15px">Data quality log</h3><div class="scroll">{quality}</div>
</section>

<footer>cryptolab v{version} · generated {generated} · research only, not investment advice.</footer>
</main></body></html>
"""
