"""
20 — Publication-ready figures (4 panels, 300 DPI, colorblind-safe palette)

Figure 1: Startup Outcomes by Investor Composition  (grouped bars + sig brackets)
Figure 2: Cross-Community Bridging and Startup Failure  (dual-axis line + CI)
Figure 3: Policy Uncertainty Fragments Investor Networks (time-series + path diagram)
Figure 4: Investor Type × Startup Novelty  (4-line chart)

Colorblind palette: Wong (2011) — green=#009E73, red/vermillion=#D55E00,
  blue=#0072B2, orange=#E69F00, grey=#999999, sky blue=#56B4E9
"""

import warnings; warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
import matplotlib.patheffects as pe
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from scipy.stats import norm, chi2_contingency

ROOT = Path(__file__).parent
OUT  = ROOT / "output" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# ── Colorblind-safe palette (Wong 2011) ─────────────────────────────────────
C = {
    "green":   "#009E73",
    "red":     "#D55E00",
    "blue":    "#0072B2",
    "orange":  "#E69F00",
    "grey":    "#999999",
    "sky":     "#56B4E9",
    "purple":  "#CC79A7",
    "black":   "#000000",
    "bg":      "#FFFFFF",
}

# ── Shared style ─────────────────────────────────────────────────────────────
FONT_TITLE  = 13
FONT_AXIS   = 11
FONT_TICK   = 10
FONT_ANNOT  = 9.5
FONT_SMALL  = 8.5
DPI         = 300
LINEWIDTH   = 2.0
ALPHA_FILL  = 0.18

plt.rcParams.update({
    "font.family":         "sans-serif",
    "font.size":           FONT_TICK,
    "axes.spines.top":     False,
    "axes.spines.right":   False,
    "axes.grid":           True,
    "grid.color":          "#e0e0e0",
    "grid.linestyle":      "-",
    "grid.linewidth":      0.6,
    "axes.axisbelow":      True,
    "figure.facecolor":    C["bg"],
    "axes.facecolor":      C["bg"],
})


# ════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ════════════════════════════════════════════════════════════════════════════

def proportion_ci(n_successes, n_total, z=1.96):
    """Wilson score confidence interval."""
    if n_total == 0:
        return 0, 0, 0
    p = n_successes / n_total
    denom = 1 + z**2 / n_total
    centre = (p + z**2 / (2 * n_total)) / denom
    margin = z * np.sqrt(p * (1 - p) / n_total + z**2 / (4 * n_total**2)) / denom
    return p * 100, max(0, centre - margin) * 100, (centre + margin) * 100


def chi2_p(n_a, N_a, n_b, N_b):
    """Chi-squared p-value for 2×2 contingency table of proportions."""
    table = [[n_a, N_a - n_a], [n_b, N_b - n_b]]
    _, p, _, _ = chi2_contingency(table, correction=False)
    return p


def sig_label(p):
    if p < 0.001: return "***"
    if p < 0.010: return "**"
    if p < 0.050: return "*"
    if p < 0.100: return "†"
    return "n.s."


def sig_bracket(ax, x1, x2, y, dy, label, colour="black", fontsize=FONT_ANNOT):
    """Draw a significance bracket between x1 and x2 at height y."""
    xs = [x1, x1, x2, x2]
    ys = [y, y + dy, y + dy, y]
    ax.plot(xs, ys, lw=1.2, color=colour, clip_on=False)
    ax.text((x1 + x2) / 2, y + dy * 1.05, label,
            ha="center", va="bottom", fontsize=fontsize, color=colour)


def despine(ax, keep=("bottom", "left")):
    for spine in ("top", "bottom", "left", "right"):
        ax.spines[spine].set_visible(spine in keep)


# ════════════════════════════════════════════════════════════════════════════
#  FIGURE 1: STARTUP OUTCOMES BY INVESTOR COMPOSITION
# ════════════════════════════════════════════════════════════════════════════

def figure1():
    df = pd.read_csv(ROOT / "output" / "cpu" / "cpu_company_dataset.csv")

    # 3-group classification with [10, 90] thresholds
    df["syn3"] = "Mixed"
    df.loc[df["pct_green"] >= 90, "syn3"] = "Pure Green"
    df.loc[df["pct_green"] <  10, "syn3"] = "Pure Non-Green"

    groups    = ["Pure Green", "Mixed", "Pure Non-Green"]
    group_x   = [0, 1, 2]
    bar_width = 0.32

    fig, ax = plt.subplots(figsize=(8.0, 5.8))

    exit_bars, fail_bars = [], []
    exit_data, fail_data = {}, {}

    for xi, g in zip(group_x, groups):
        grp = df[df["syn3"] == g]
        n   = len(grp)
        ne  = grp["exited"].sum()
        nf  = grp["failed"].sum()

        ep, ep_lo, ep_hi = proportion_ci(ne, n)
        fp, fp_lo, fp_hi = proportion_ci(nf, n)

        exit_data[g] = {"p": ep, "lo": ep_lo, "hi": ep_hi, "n": n, "n_ev": ne}
        fail_data[g] = {"p": fp, "lo": fp_lo, "hi": fp_hi, "n": n, "n_ev": nf}

        e_bar = ax.bar(xi - bar_width / 2, ep, bar_width,
                       color=C["green"], label="Exit rate" if xi == 0 else "_",
                       alpha=0.88, zorder=3, edgecolor="white", linewidth=0.8)
        f_bar = ax.bar(xi + bar_width / 2, fp, bar_width,
                       color=C["red"], label="Failure rate" if xi == 0 else "_",
                       alpha=0.88, zorder=3, edgecolor="white", linewidth=0.8)

        ax.errorbar(xi - bar_width / 2, ep,
                    yerr=[[ep - ep_lo], [ep_hi - ep]],
                    fmt="none", color="black", capsize=4, linewidth=1.2, zorder=4)
        ax.errorbar(xi + bar_width / 2, fp,
                    yerr=[[fp - fp_lo], [fp_hi - fp]],
                    fmt="none", color="black", capsize=4, linewidth=1.2, zorder=4)

        # n= annotation below x-tick
        ax.text(xi, -1.6, f"n={n:,}", ha="center", va="top",
                fontsize=FONT_SMALL, color="#555555")

        exit_bars.append(e_bar); fail_bars.append(f_bar)

    # ── Significance brackets (failure rates) ────────────────────────────────
    y_top = 18.0  # headroom for brackets
    ax.set_ylim(-1.8, y_top + 1)

    def fail_bracket(g1, g2, y_level):
        d1, d2 = fail_data[g1], fail_data[g2]
        p = chi2_p(d1["n_ev"], d1["n"], d2["n_ev"], d2["n"])
        lbl = sig_label(p)
        x1 = groups.index(g1)
        x2 = groups.index(g2)
        sig_bracket(ax, x1 + bar_width / 2, x2 + bar_width / 2,
                    y_level, 0.7, lbl + f"  (p={p:.3f})", colour=C["red"])

    fail_bracket("Mixed", "Pure Green",     14.5)
    fail_bracket("Mixed", "Pure Non-Green", 16.5)

    # ── Axes ─────────────────────────────────────────────────────────────────
    ax.set_xticks(group_x)
    ax.set_xticklabels(["Pure Green\n(≥90% green)", "Mixed\n(10–90% green)",
                         "Pure Non-Green\n(<10% green)"],
                       fontsize=FONT_TICK)
    ax.set_ylabel("Rate (%)", fontsize=FONT_AXIS)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.set_xlim(-0.55, 2.55)
    ax.set_xlabel("")
    despine(ax)

    legend = ax.legend(fontsize=FONT_TICK, frameon=False, loc="upper left",
                       ncol=2, handlelength=1.4, handletextpad=0.5)

    ax.set_title("Startup Outcomes by Investor Composition",
                 fontsize=FONT_TITLE, fontweight="bold", pad=12)

    # Finding callout
    ax.text(1.0, 12.8,
            "Mixed syndicates have the lowest failure rate\n"
            "(7.8% vs 9.5% pure green, 14.4% non-green)",
            ha="center", va="bottom", fontsize=FONT_SMALL,
            style="italic", color="#444444",
            bbox=dict(facecolor="#f9f9f9", edgecolor="#cccccc",
                      boxstyle="round,pad=0.4", alpha=0.9))

    fig.tight_layout()
    path = OUT / "figure1_syndicate_outcomes.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor=C["bg"])
    plt.close(fig)
    print(f"  Saved {path.name}")


# ════════════════════════════════════════════════════════════════════════════
#  FIGURE 2: CROSS-COMMUNITY DOSE-RESPONSE
# ════════════════════════════════════════════════════════════════════════════

def figure2():
    df = pd.read_csv(ROOT / "output" / "cpu" / "cpu_company_dataset.csv")
    df = df[df["n_communities"] >= 1].copy()  # exclude 0-community (unmatched)

    df["n_comm_cat"]   = df["n_communities"].apply(
        lambda x: "1" if x == 1 else "2" if x == 2 else "3+"
    )
    cats = ["1", "2", "3+"]
    cat_x = np.arange(len(cats))

    fail_pts, fail_lo, fail_hi = [], [], []
    cap_pts,  cap_lo,  cap_hi  = [], [], []
    n_pts = []

    for cat in cats:
        g = df[df["n_comm_cat"] == cat]
        n = len(g)
        nf = g["failed"].sum()
        fp, flo, fhi = proportion_ci(nf, n)
        fail_pts.append(fp); fail_lo.append(flo); fail_hi.append(fhi)

        med = g["Total Raised"].median()
        # bootstrap CI for median
        boots = [g["Total Raised"].sample(n, replace=True).median()
                 for _ in range(500)]
        cap_pts.append(med)
        cap_lo.append(np.percentile(boots, 2.5))
        cap_hi.append(np.percentile(boots, 97.5))
        n_pts.append(n)

    # ── p-value for linear dose-response (Cochran-Armitage) ─────────────────
    fail_arr = np.array([df[df["n_comm_cat"] == c]["failed"].values for c in cats], dtype=object)
    scores   = np.array([1, 2, 3])
    ns       = np.array(n_pts)
    ps       = np.array([fp / 100 for fp in fail_pts])
    p_bar    = (df["failed"].sum()) / len(df)
    T_arm    = sum(scores[i] * ns[i] * (ps[i] - p_bar) for i in range(3))
    V_arm    = p_bar * (1 - p_bar) * sum(ns[i] * (scores[i] - (ns*scores).sum()/ns.sum())**2 for i in range(3))
    z_arm    = T_arm / np.sqrt(V_arm) if V_arm > 0 else 0
    p_arm    = 2 * (1 - norm.cdf(abs(z_arm)))

    # ── Figure ───────────────────────────────────────────────────────────────
    fig, ax1 = plt.subplots(figsize=(7.5, 5.5))
    ax2 = ax1.twinx()

    # Failure rate line (left axis)
    ax1.fill_between(cat_x, fail_lo, fail_hi,
                     color=C["red"], alpha=ALPHA_FILL, zorder=2)
    ax1.plot(cat_x, fail_pts, "-o", color=C["red"],
             linewidth=LINEWIDTH, markersize=8, label="Failure rate (%)", zorder=4)
    ax1.errorbar(cat_x, fail_pts,
                 yerr=[np.array(fail_pts) - np.array(fail_lo),
                       np.array(fail_hi)  - np.array(fail_pts)],
                 fmt="none", color=C["red"], capsize=5, linewidth=1.2, zorder=5)

    # Capital line (right axis)
    ax2.fill_between(cat_x, cap_lo, cap_hi,
                     color=C["blue"], alpha=ALPHA_FILL, zorder=2)
    ax2.plot(cat_x, cap_pts, "--s", color=C["blue"],
             linewidth=LINEWIDTH, markersize=8, label="Median capital raised ($M)", zorder=4)
    ax2.errorbar(cat_x, cap_pts,
                 yerr=[np.array(cap_pts) - np.array(cap_lo),
                       np.array(cap_hi)  - np.array(cap_pts)],
                 fmt="none", color=C["blue"], capsize=5, linewidth=1.2, zorder=5)

    # Axes formatting
    ax1.set_xticks(cat_x)
    ax1.set_xticklabels([f"{c}\n(n={n:,})" for c, n in zip(cats, n_pts)],
                        fontsize=FONT_TICK)
    ax1.set_xlabel("Number of Investor Communities Spanned", fontsize=FONT_AXIS)
    ax1.set_ylabel("Failure Rate (%)", color=C["red"], fontsize=FONT_AXIS)
    ax1.tick_params(axis="y", labelcolor=C["red"])
    ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
    ax1.set_ylim(0, max(fail_hi) * 1.35)

    ax2.set_ylabel("Median Capital Raised ($M)", color=C["blue"], fontsize=FONT_AXIS)
    ax2.tick_params(axis="y", labelcolor=C["blue"])
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.1f M"))
    ax2.set_ylim(0, max(cap_hi) * 1.6)

    # Annotation box
    pstr = f"p<0.001" if p_arm < 0.001 else f"p={p_arm:.3f}"
    ax1.text(0.03, 0.97,
             f"Cochran-Armitage trend test\n{pstr}  (N={len(df):,})",
             transform=ax1.transAxes, va="top", ha="left", fontsize=FONT_SMALL,
             bbox=dict(facecolor="#f9f9f9", edgecolor="#cccccc",
                       boxstyle="round,pad=0.4", alpha=0.9))

    # Combined legend
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, fontsize=FONT_TICK, frameon=False,
               loc="upper right")

    ax1.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax1.grid(True, linestyle="-", linewidth=0.6, color="#e0e0e0", zorder=1)
    ax2.grid(False)

    ax1.set_title("Cross-Community Bridging and Startup Failure",
                  fontsize=FONT_TITLE, fontweight="bold", pad=12)

    fig.tight_layout()
    path = OUT / "figure2_crosscommunity_doseresponse.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor=C["bg"])
    plt.close(fig)
    print(f"  Saved {path.name}")


# ════════════════════════════════════════════════════════════════════════════
#  FIGURE 3: CPU → NETWORK FRAGMENTATION (TIME SERIES + PATH DIAGRAM)
# ════════════════════════════════════════════════════════════════════════════

def figure3():
    q_ts = pd.read_csv(ROOT / "output" / "cpu_dynamics" / "quarterly_composition.csv")
    q_ts["date"] = pd.to_datetime(
        q_ts["quarter"].str.replace(r"-Q(\d)", lambda m: f"-{(int(m.group(1))-1)*3+1:02d}-01",
                                    regex=True)
    )
    q_ts = q_ts.sort_values("date").reset_index(drop=True)

    fig, (ax_ts, ax_path) = plt.subplots(
        1, 2, figsize=(14, 5.5),
        gridspec_kw={"width_ratios": [1.55, 1]}
    )
    fig.suptitle("Policy Uncertainty Fragments Investor Networks",
                 fontsize=FONT_TITLE, fontweight="bold", y=1.01)

    # ── Panel A: Time Series ─────────────────────────────────────────────────
    # Detrend both series so secular trends don't dominate
    t = np.arange(len(q_ts))
    def detrend_zscore(s):
        valid = s.dropna()
        idx   = s.dropna().index
        trend = np.polyfit(t[idx], valid, 1)
        resid = valid - np.polyval(trend, t[idx])
        return (resid - resid.mean()) / resid.std()

    cpu_z_dt  = detrend_zscore(q_ts["cpu_narrow"])
    cross_z_dt = detrend_zscore(q_ts["pct_cross"])

    # Align on common index
    common_idx = cpu_z_dt.index.intersection(cross_z_dt.index)
    dates_c    = q_ts.loc[common_idx, "date"]

    # 1-quarter lag on cross (shift forward 1 quarter to show delayed response)
    cross_lagged = cross_z_dt.shift(-1)  # show what happens 1Q after CPU
    cross_lagged_aligned = cross_lagged.loc[common_idx]

    ax_ts.fill_between(dates_c, cpu_z_dt.loc[common_idx],
                       alpha=0.18, color=C["blue"])
    l1, = ax_ts.plot(dates_c, cpu_z_dt.loc[common_idx],
                     lw=LINEWIDTH, color=C["blue"], label="CPU Index (detrended z-score)")
    l2, = ax_ts.plot(dates_c, cross_lagged_aligned,
                     lw=LINEWIDTH, color=C["green"], ls="--",
                     label="% Cross-community deals\n(1Q later, detrended z-score)")
    ax_ts.fill_between(dates_c, cross_lagged_aligned,
                       alpha=0.12, color=C["green"])

    # Policy event markers
    events = [("2017-01-01","Trump I",C["red"]),("2019-12-01","EU Green Deal",C["green"]),
              ("2022-08-01","IRA",C["green"]),("2025-01-01","Trump II",C["red"])]
    ylim_ts = ax_ts.get_ylim()
    for dt_str, label, ec in events:
        dt = pd.Timestamp(dt_str)
        if dates_c.min() <= dt <= dates_c.max():
            ax_ts.axvline(dt, color=ec, lw=1.2, ls=":", alpha=0.8, zorder=1)
            ax_ts.text(dt, 2.5, label, rotation=90, fontsize=7.5,
                       va="top", ha="right", color=ec, fontweight="bold")

    ax_ts.axhline(0, color="black", lw=0.7, ls="-", alpha=0.4)
    ax_ts.set_ylabel("Standardised Value (z-score)", fontsize=FONT_AXIS)
    ax_ts.set_xlabel("Quarter", fontsize=FONT_AXIS)
    ax_ts.legend(fontsize=FONT_SMALL, frameon=False, loc="lower left")
    ax_ts.set_title("Panel A: CPU vs Cross-Community Deal Share",
                    fontsize=FONT_AXIS, fontweight="bold")
    ax_ts.spines["top"].set_visible(False)
    ax_ts.spines["right"].set_visible(False)
    ax_ts.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax_ts.xaxis.set_major_locator(mdates.YearLocator(2))
    plt.setp(ax_ts.xaxis.get_majorticklabels(), rotation=30, ha="right")
    ax_ts.grid(True, linestyle="-", linewidth=0.6, color="#e0e0e0")

    # ── Panel B: Path Diagram ────────────────────────────────────────────────
    ax_path.set_xlim(0, 10)
    ax_path.set_ylim(0, 9)
    ax_path.axis("off")
    ax_path.set_title("Panel B: Mediation Path Diagram",
                      fontsize=FONT_AXIS, fontweight="bold")

    # Coefficients from Script 19 mediation (individual-level bootstrap)
    A_coef, A_p  = "+0.487", "<0.001"   # CPU → quarterly cross-comm share
    B_coef, B_p  = "−0.014", "<0.001"   # quarterly cross-comm → failure
    C_direct     = "p = 0.89, n.s."     # CPU → failure (from Script 18, controlled)
    indirect_txt = "Indirect: −0.007\n95% CI [−0.010, −0.003]"
    boot_txt     = "Bootstrap p < 0.001\nPartial mediation"

    # Node positions & styles
    node_specs = {
        "cpu":   (2.0, 5.5, "CPU Index\n(Gavriilidis)",   C["blue"]),
        "med":   (5.0, 8.2, "Quarterly\nCross-Community\nDeal Share",  C["green"]),
        "out":   (8.0, 5.5, "Startup\nFailure Rate",    C["red"]),
    }

    def node(ax, x, y, text, colour, radius=0.85):
        ax.add_patch(mpatches.FancyBboxPatch(
            (x - radius, y - 0.65), 2*radius, 1.3,
            boxstyle="round,pad=0.15", facecolor=colour, alpha=0.85,
            edgecolor="white", linewidth=2, zorder=5
        ))
        ax.text(x, y, text, ha="center", va="center",
                fontsize=8, fontweight="bold", color="white", zorder=6,
                multialignment="center")

    for key, (x, y, lbl, col) in node_specs.items():
        node(ax_path, x, y, lbl, col)

    def arrow(ax, x1, y1, x2, y2, text, text_y_off, colour, lw=2.2):
        ax.annotate("",
            xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="-|>", color=colour, lw=lw,
                            connectionstyle="arc3,rad=0.0"),
            zorder=4)
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx, my + text_y_off, text, ha="center", va="bottom",
                fontsize=8.5, color=colour, fontweight="bold",
                bbox=dict(facecolor="white", alpha=0.8,
                          edgecolor="none", pad=1.5))

    # Path A: CPU → cross-community (green)
    arrow(ax_path, 3.0, 6.0, 4.1, 7.7,
          f"Path A: β={A_coef}***", 0.15, C["green"])

    # Path B: cross-community → failure (red)
    arrow(ax_path, 5.9, 7.7, 6.9, 6.1,
          f"Path B: β={B_coef}***", 0.15, C["red"])

    # Direct path (crossed out, grey dashed)
    ax_path.annotate("",
        xy=(7.1, 5.5), xytext=(2.9, 5.5),
        arrowprops=dict(arrowstyle="-|>", color=C["grey"], lw=1.5, ls="dashed"),
        zorder=3)
    ax_path.text(5.0, 5.1,
                 f"Direct path: {C_direct}", ha="center", va="top",
                 fontsize=8, color=C["grey"], style="italic",
                 bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=1))
    # Strike-through line over direct path
    ax_path.plot([3.3, 6.7], [5.55, 5.55], color=C["grey"], lw=3.5, alpha=0.7, zorder=4)

    # Indirect effect annotation box
    ax_path.text(5.0, 1.8,
                 indirect_txt + "\n" + boot_txt,
                 ha="center", va="center", fontsize=8.5, color="black",
                 fontweight="bold",
                 bbox=dict(facecolor="#f0f8f0", edgecolor=C["green"],
                           boxstyle="round,pad=0.5", linewidth=1.5))

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    path = OUT / "figure3_cpu_mediation.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor=C["bg"])
    plt.close(fig)
    print(f"  Saved {path.name}")


# ════════════════════════════════════════════════════════════════════════════
#  FIGURE 4: INVESTOR TYPE × STARTUP NOVELTY
# ════════════════════════════════════════════════════════════════════════════

def figure4():
    df = pd.read_csv(ROOT / "output" / "cpu" / "cpu_company_dataset.csv")

    # Define green/non-green cleanly
    df["is_green"]     = (df["pct_green"] >= 50).astype(int)
    df["is_non_green"] = (df["pct_green"] <  20).astype(int)
    df_clean           = df[(df["is_green"] == 1) | (df["is_non_green"] == 1)].copy()

    # Novelty quartiles
    df_clean["nov_q"] = pd.qcut(df_clean["specter_novelty"], q=4,
                                labels=["Q1\n(Conventional)", "Q2", "Q3",
                                        "Q4\n(Novel)"])

    x_vals = [0, 1, 2, 3]
    x_labels = ["Q1\n(Conventional)", "Q2", "Q3", "Q4\n(Novel)"]
    quartile_labels = df_clean["nov_q"].cat.categories.tolist()

    # Accumulate data
    data = {
        "green_exit": [], "green_fail": [], "nong_exit": [], "nong_fail": [],
        "green_exit_lo": [], "green_exit_hi": [],
        "green_fail_lo": [], "green_fail_hi": [],
        "nong_exit_lo": [],  "nong_exit_hi": [],
        "nong_fail_lo": [],  "nong_fail_hi": [],
        "n_green": [],       "n_nong": [],
    }

    for q in quartile_labels:
        g_g  = df_clean[(df_clean["nov_q"] == q) & (df_clean["is_green"] == 1)]
        g_ng = df_clean[(df_clean["nov_q"] == q) & (df_clean["is_non_green"] == 1)]

        for g, prefix in [(g_g, "green"), (g_ng, "nong")]:
            ne = g["exited"].sum(); nf = g["failed"].sum(); n = len(g)
            ep, elo, ehi = proportion_ci(ne, n)
            fp, flo, fhi = proportion_ci(nf, n)
            data[f"{prefix}_exit"].append(ep)
            data[f"{prefix}_fail"].append(fp)
            data[f"{prefix}_exit_lo"].append(elo)
            data[f"{prefix}_exit_hi"].append(ehi)
            data[f"{prefix}_fail_lo"].append(flo)
            data[f"{prefix}_fail_hi"].append(fhi)
            data[f"n_{prefix}"].append(n)

    fig, ax = plt.subplots(figsize=(8.5, 5.8))

    kw_ci = dict(alpha=0.14, zorder=2)

    # Pure Green exit (solid green)
    ax.fill_between(x_vals, data["green_exit_lo"], data["green_exit_hi"],
                    color=C["green"], **kw_ci)
    l1, = ax.plot(x_vals, data["green_exit"], "-o", color=C["green"],
                  lw=LINEWIDTH, ms=8, label="Green-backed: exit rate", zorder=4)

    # Pure Green failure (dashed green)
    ax.fill_between(x_vals, data["green_fail_lo"], data["green_fail_hi"],
                    color=C["green"], **kw_ci)
    l2, = ax.plot(x_vals, data["green_fail"], "--o", color=C["green"],
                  lw=LINEWIDTH, ms=8, markerfacecolor="white",
                  markeredgewidth=2, label="Green-backed: failure rate", zorder=4)

    # Pure Non-Green exit (solid grey)
    ax.fill_between(x_vals, data["nong_exit_lo"], data["nong_exit_hi"],
                    color=C["grey"], **kw_ci)
    l3, = ax.plot(x_vals, data["nong_exit"], "-^", color=C["grey"],
                  lw=LINEWIDTH, ms=8, label="Non-green-backed: exit rate", zorder=4)

    # Pure Non-Green failure (dashed grey/red)
    ax.fill_between(x_vals, data["nong_fail_lo"], data["nong_fail_hi"],
                    color=C["red"], alpha=0.10, zorder=2)
    l4, = ax.plot(x_vals, data["nong_fail"], "--^", color=C["red"],
                  lw=LINEWIDTH, ms=8, markerfacecolor="white",
                  markeredgewidth=2, label="Non-green-backed: failure rate", zorder=4)

    # Annotations at Q4
    q4_fail_g  = data["green_fail"][-1]
    q4_fail_ng = data["nong_fail"][-1]
    q1_fail_ng = data["nong_fail"][0]

    ax.annotate(f"Green: {q4_fail_g:.1f}%\n(flat)",
                xy=(3, q4_fail_g), xytext=(2.55, q4_fail_g - 2.0),
                fontsize=FONT_SMALL, color=C["green"], fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=C["green"], lw=0.8),
                ha="right")
    ax.annotate(f"Non-green: {q4_fail_ng:.1f}%\n(+{q4_fail_ng-q1_fail_ng:.1f}pp from Q1)",
                xy=(3, q4_fail_ng), xytext=(2.55, q4_fail_ng + 1.5),
                fontsize=FONT_SMALL, color=C["red"], fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=C["red"], lw=0.8),
                ha="right")

    # Q4 divergence marker
    ax.annotate("", xy=(3, q4_fail_g), xytext=(3, q4_fail_ng),
                arrowprops=dict(arrowstyle="<->", color="#555555", lw=1.4))
    ax.text(3.08, (q4_fail_g + q4_fail_ng)/2,
            f"Δ {q4_fail_ng - q4_fail_g:.1f}pp", fontsize=FONT_SMALL,
            color="#333333", va="center")

    # n= under each x-tick
    for xi, (ng, nng) in enumerate(zip(data["n_green"], data["n_nong"])):
        ax.text(xi, -1.5, f"Green n={ng:,}\nNon-green n={nng:,}",
                ha="center", va="top", fontsize=7.5, color="#555555")

    ax.set_xticks(x_vals)
    ax.set_xticklabels(x_labels, fontsize=FONT_TICK)
    ax.set_xlabel("Startup Novelty Quartile (Specter embeddings)", fontsize=FONT_AXIS)
    ax.set_ylabel("Rate (%)", fontsize=FONT_AXIS)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.set_ylim(-2, max(data["nong_fail_hi"]) * 1.25)
    ax.legend(fontsize=FONT_TICK, frameon=False, loc="upper left",
              ncol=2, handlelength=1.8)
    despine(ax)

    ax.set_title("Investor Type × Startup Novelty",
                 fontsize=FONT_TITLE, fontweight="bold", pad=12)

    fig.tight_layout()
    path = OUT / "figure4_novelty_syndicate.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor=C["bg"])
    plt.close(fig)
    print(f"  Saved {path.name}")


# ════════════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Generating publication-ready figures...")
    figure1()
    figure2()
    figure3()
    figure4()
    print(f"\nAll figures saved to {OUT}")
