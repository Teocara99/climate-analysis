"""
Analysis 8 — Four survival/success models comparing Pure Green, Mixed, and
Pure Non-Green investor portfolios.

Groups (based on share of GREEN_VC-classified investors):
  Pure Green    : pct_green >= 75%
  Mixed         : 25% < pct_green < 75%
  Pure Non-Green: pct_green <= 25%

Model 1 - Survival      : company survived (operating OR exited via IPO/M&A)
Model 2 - Exit Success  : exited via IPO or M&A (literature standard)
Model 3 - Failure        : company failed/bankrupt/closed
Model 4 - Kaplan-Meier  : time-to-failure curves (censored at last known date)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import numpy as np
from scipy import stats
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test
from pathlib import Path
from load_data import load_companies

OUT = Path(__file__).parent / "output"
CENSORING_DATE = pd.Timestamp("2026-07-03")

GROUP_COLORS  = {
    "Pure Green\n(≥75%)":     "#27ae60",
    "Mixed\n(25-75%)":        "#f39c12",
    "Pure Non-Green\n(≤25%)": "#7f8c8d",
}
GROUP_ORDER = ["Pure Green\n(≥75%)", "Mixed\n(25-75%)", "Pure Non-Green\n(≤25%)"]


# ── Outcome helpers ────────────────────────────────────────────────────────
def outcome(row) -> str:
    own = str(row.get("Ownership Status", ""))
    biz = str(row.get("Business Status", ""))
    if "Publicly Held" in own or "IPO" in own:
        return "ipo"
    if "Acquired" in own:
        return "acquired"
    if "Out of Business" in own or "Liquidation" in biz or \
       "Out of Business" in biz or "Reorg" in biz:
        return "failed"
    return "operating"


def assign_group(pct_green: float) -> str:
    if pct_green >= 75:
        return "Pure Green\n(≥75%)"
    elif pct_green <= 25:
        return "Pure Non-Green\n(≤25%)"
    else:
        return "Mixed\n(25-75%)"


def chi2_pval(df: pd.DataFrame, col: str) -> str:
    ct = pd.crosstab(df["group"], df[col])
    chi2, p, *_ = stats.chi2_contingency(ct)
    return f"χ²={chi2:.1f}, p={p:.3f}"


# ── Load & merge ───────────────────────────────────────────────────────────
def build_df() -> pd.DataFrame:
    companies  = load_companies()
    mix        = pd.read_csv(OUT / "company_investor_mix.csv")

    df = companies.merge(mix[["Company ID", "pct_green", "n_investors_matched"]],
                         on="Company ID", how="inner")
    df = df[df["n_investors_matched"] >= 2].copy()
    df["outcome_str"]   = df.apply(outcome, axis=1)
    df["group"]         = df["pct_green"].apply(assign_group)

    # Binary labels
    df["survived"]      = (df["outcome_str"].isin(["ipo", "acquired", "operating"])).astype(int)
    df["exited"]        = (df["outcome_str"].isin(["ipo", "acquired"])).astype(int)
    df["failed"]        = (df["outcome_str"] == "failed").astype(int)

    # Survival time (years from first financing to last known date / event)
    df["t_start"] = pd.to_datetime(df["First Financing Date"], errors="coerce")
    df["t_end"]   = pd.to_datetime(df["Last Financing Date"],  errors="coerce").fillna(CENSORING_DATE)
    df["t_end"]   = df["t_end"].clip(upper=CENSORING_DATE)
    df["duration"] = (df["t_end"] - df["t_start"]).dt.days / 365.25

    # For KM: event = failed (1) or still at risk (0 = censored)
    df["km_event"]    = df["failed"]
    df["km_duration"] = df["duration"].clip(lower=0.01)  # no zero durations

    df = df.dropna(subset=["t_start", "km_duration"])
    return df


def bar_chart(ax, df: pd.DataFrame, col: str, title: str, ylabel: str, ymax: float = None):
    rates  = df.groupby("group")[col].mean() * 100
    counts = df.groupby("group").size()
    rates  = rates.reindex(GROUP_ORDER)
    counts = counts.reindex(GROUP_ORDER)
    colors = [GROUP_COLORS[g] for g in GROUP_ORDER]
    bars = ax.bar(GROUP_ORDER, rates.values, color=colors, edgecolor="white", width=0.55)
    for bar, rate, n in zip(bars, rates.values, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                f"{rate:.1f}%\n(n={n:,})", ha="center", va="bottom", fontsize=9)
    p_str = chi2_pval(df, col)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, (ymax or rates.max() * 1.3))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.text(0.98, 0.96, p_str, transform=ax.transAxes, ha="right", va="top",
            fontsize=8, color="dimgrey",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="lightgrey"))


def main():
    df = build_df()
    print(f"Companies in analysis: {len(df):,}")
    print(df.groupby("group")[["survived","exited","failed"]].agg(["mean","sum"]).round(3))

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle(
        "Investor Green Mix vs Company Outcomes — Climate Tech 2012–2026\n"
        "Groups: % of classified investors with GREEN_VC focus",
        fontsize=13, fontweight="bold", y=1.01
    )

    # ── Model 1: Survival ─────────────────────────────────────────────────
    bar_chart(axes[0, 0], df, "survived",
              "Model 1 — Survival\n(operating + IPO + M&A = success)",
              "Survival Rate (%)", ymax=105)
    axes[0, 0].annotate("Basis: Reuther 2025, Manigart 2002",
                         xy=(0.01, 0.03), xycoords="axes fraction",
                         fontsize=7, color="grey")

    # ── Model 2: Exit Success ─────────────────────────────────────────────
    bar_chart(axes[0, 1], df, "exited",
              "Model 2 — Exit Success\n(IPO or M&A = success; still operating = failure)",
              "Exit Rate (%)")
    axes[0, 1].annotate("Basis: Cumming 2017, Hochberg 2007",
                         xy=(0.01, 0.03), xycoords="axes fraction",
                         fontsize=7, color="grey")

    # ── Model 3: Failure Avoidance ────────────────────────────────────────
    bar_chart(axes[1, 0], df, "failed",
              "Model 3 — Failure Rate\n(bankrupt / closed = failure)",
              "Failure Rate (%)")
    axes[1, 0].annotate("Model 1 flipped — framing matters for interpretation",
                         xy=(0.01, 0.03), xycoords="axes fraction",
                         fontsize=7, color="grey")

    # ── Model 4: Kaplan-Meier ─────────────────────────────────────────────
    ax = axes[1, 1]
    kmf = KaplanMeierFitter()
    lrt_data = {}
    for grp in GROUP_ORDER:
        sub = df[df["group"] == grp]
        kmf.fit(sub["km_duration"], event_observed=sub["km_event"], label=grp)
        kmf.plot_survival_function(
            ax=ax, ci_show=True, ci_alpha=0.1,
            color=GROUP_COLORS[grp], linewidth=2
        )
        lrt_data[grp] = sub

    # Log-rank test (pairwise between Pure Green vs Pure Non-Green)
    pg  = df[df["group"] == "Pure Green\n(≥75%)"]
    png = df[df["group"] == "Pure Non-Green\n(≤25%)"]
    lrt = logrank_test(pg["km_duration"],  png["km_duration"],
                       pg["km_event"],     png["km_event"])

    ax.set_title("Model 4 — Kaplan-Meier Failure Survival\n"
                 "(event = company failure; censored = still operating)",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Years from First Investment")
    ax.set_ylabel("P(Survival) — Failure-Free Probability")
    ax.set_ylim(0.6, 1.02)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.grid(linestyle="--", alpha=0.4)
    ax.text(0.98, 0.96,
            f"Log-rank (Green vs Non-Green):\np={lrt.p_value:.3f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=8,
            color="dimgrey",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="lightgrey"))
    ax.annotate("Basis: Pommet 2011, Manigart 2002 — censored: still operating",
                 xy=(0.01, 0.03), xycoords="axes fraction",
                 fontsize=7, color="grey")
    ax.legend(loc="lower left", fontsize=9)

    plt.tight_layout()
    fig.savefig(OUT / "survival_models_4panel.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved → {OUT / 'survival_models_4panel.png'}")

    # ── Summary table ─────────────────────────────────────────────────────
    summary = df.groupby("group").agg(
        n=("survived", "count"),
        survival_rate=("survived", lambda x: f"{x.mean()*100:.1f}%"),
        exit_rate=("exited",  lambda x: f"{x.mean()*100:.1f}%"),
        failure_rate=("failed", lambda x: f"{x.mean()*100:.1f}%"),
        median_duration=("km_duration", "median"),
    ).reindex(GROUP_ORDER)
    print("\n--- Summary by group ---")
    print(summary.to_string())


if __name__ == "__main__":
    main()
