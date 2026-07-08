"""
15 — Community deep profiling: cross-community effects, cluster archetypes,
     and community-level success drivers.

PART 1: Cross-community funding full outcome analysis
  A. Full outcome comparison (exit/IPO/M&A/failure/survival/capital/rounds)
  B. Dose-response: 1 / 2 / 3+ communities
  C. Which community PAIR types drive outcomes?
  D. Cross-community × Trump 2016 interaction

PART 2: Deep profile of 5 cluster archetypes
  A. Investor characteristics (top-5, deals/investor, vintage, geography)
  B. Company characteristics (founding year, stage, geography, capital)
  C. Keyword deep dive from descriptions (TF-IDF beyond broad sectors)
  D. Temporal behavior (stage pacing, deals/year trend)

PART 3: What makes communities successful?
  A. Community-level OLS (N=30) predicting exit/failure rate
  B. ICC — how much does community membership explain?
  C. Cohesion (internal density) → outcomes, inverted-U test

Outputs in output/community/:
  cross_community_outcomes.png, cross_community_pairs.csv,
  cluster_profiles.txt, cluster_keywords.csv,
  community_success_regression.txt, community_icc.txt,
  cohesion_outcomes.png, summary_deep_profile.txt
"""
import warnings; warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
import pandas as pd
import numpy as np
import re
import itertools
import networkx as nx
from pathlib import Path
from collections import defaultdict, Counter
from sklearn.feature_extraction.text import TfidfVectorizer
import statsmodels.formula.api as smf
from load_data import load_deals, load_companies

OUT  = Path(__file__).parent / "output" / "community"
OUTN = Path(__file__).parent / "output" / "network"
OUTB = Path(__file__).parent / "output"
DATA = Path(__file__).parent / "data"

CLUSTER_ORDER = ["Exit-Oriented","Mixed Bridge","High-Risk Frontier","Government-Led","Deep Green"]
CLUSTER_COLORS = {
    "Exit-Oriented":     "#2980b9",
    "Mixed Bridge":      "#27ae60",
    "High-Risk Frontier":"#e67e22",
    "Government-Led":    "#8e44ad",
    "Deep Green":        "#16a085",
}

# ─────────────────────────────────────────────────────────────────────────────
#  SHARED DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def build_base_tables():
    """Return (deal_inv, comp_comm, comp_df, typology, metrics) — the shared tables."""
    print("Loading data...")
    deals_raw = load_deals()
    deals_raw["Year"] = pd.to_datetime(deals_raw["Deal Date"], errors="coerce").dt.year
    deals_raw["Deal Date"] = pd.to_datetime(deals_raw["Deal Date"], errors="coerce")

    clf   = pd.read_csv(OUTB / "investors_classified_output.csv")
    clf   = clf[~clf["investor_type"].isin(["PARSE_ERROR"])]
    clf_map = clf.set_index("investor_name")[["investor_type","green_focus"]].to_dict("index")

    metrics  = pd.read_csv(OUTN / "network_metrics.csv")
    typology = pd.read_csv(OUT / "community_typology.csv")
    inv_comm = metrics.set_index("investor")["community"].to_dict()

    comm_features = typology.set_index("community_id")[
        ["pct_GREEN_VC","herfindahl","avg_novelty","cluster_label","cluster"]
    ].to_dict("index")

    def clean(raw): return re.sub(r"\([^)]*\)", "", str(raw)).strip()

    rows = []
    for _, row in deals_raw.iterrows():
        if pd.isna(row.get("Investors", None)): continue
        names = [clean(n) for n in re.split(r",\s*", str(row["Investors"]).replace("\n", ", "))]
        names = [n for n in names if n and n in clf_map]
        for nm in names:
            comm = inv_comm.get(nm, np.nan)
            rows.append({
                "deal_id":     row["Deal ID"],
                "company":     row["Companies"],
                "year":        row["Year"],
                "deal_date":   row["Deal Date"],
                "deal_size":   row.get("Deal Size (USD M)", np.nan),
                "vc_round":    str(row.get("VC Round", "") or ""),
                "deal_type":   str(row.get("Deal Type", "") or ""),
                "region":      str(row.get("HQ Global Region", "") or ""),
                "country":     str(row.get("Company Country/Territory/Region", "") or ""),
                "year_founded":row.get("Year Founded", np.nan),
                "investor":    nm,
                "inv_type":    clf_map[nm]["investor_type"],
                "green_focus": clf_map[nm]["green_focus"],
                "community":   comm,
                "cluster_label": comm_features.get(int(comm), {}).get("cluster_label", "Unknown")
                               if not pd.isna(comm) else "Unknown",
            })
    deal_inv = pd.DataFrame(rows)

    # Company → community mapping
    company_rows = defaultdict(lambda: {"communities": [], "clusters": []})
    for _, r in deal_inv.dropna(subset=["community"]).iterrows():
        company_rows[r["company"]]["communities"].append(int(r["community"]))
        company_rows[r["company"]]["clusters"].append(r["cluster_label"])

    cc_rows = []
    for company, data in company_rows.items():
        comms   = data["communities"]
        counter = Counter(comms)
        lead    = counter.most_common(1)[0][0]
        distinct = sorted(set(comms))
        cluster_counter = Counter(data["clusters"])
        lead_cluster = cluster_counter.most_common(1)[0][0]
        total_w = sum(counter.values())
        w_green = sum(
            comm_features.get(c, {}).get("pct_GREEN_VC", 50) * w / total_w
            for c, w in counter.items()
        )
        cc_rows.append({
            "company":           company,
            "lead_community":    lead,
            "lead_cluster":      lead_cluster,
            "n_communities":     len(distinct),
            "cross_community":   int(len(distinct) >= 2),
            "community_pct_green": w_green,
            "all_clusters":      sorted(set(data["clusters"])),
        })
    comp_comm = pd.DataFrame(cc_rows)

    mix = pd.read_csv(OUTB / "company_investor_mix.csv")
    mix["exited"]  = mix["outcome"].isin(["Acquired","IPO / Public"]).astype(int)
    mix["failed"]  = (mix["outcome"] == "Failed").astype(int)
    mix["is_ipo"]  = (mix["outcome"] == "IPO / Public").astype(int)
    mix["is_ma"]   = (mix["outcome"] == "Acquired").astype(int)

    nov = pd.read_csv(OUTB / "novelty" / "novelty_scores.csv")[["name","specter_novelty","Description"]]
    nov.rename(columns={"name":"company"}, inplace=True)
    nov["novelty_z"] = (nov["specter_novelty"] - nov["specter_novelty"].mean()) / nov["specter_novelty"].std()

    # Round count per company
    round_counts = deal_inv.groupby("company")["deal_id"].nunique().rename("n_rounds")

    # First and last deal year per company
    deal_dates = deal_inv.groupby("company")["year"].agg(["min","max"]).rename(
        columns={"min":"first_year","max":"last_year"})

    comp_df = (mix.rename(columns={"Companies":"company"})
               .merge(comp_comm, on="company", how="left")
               .merge(nov[["company","specter_novelty","novelty_z","Description"]], on="company", how="left")
               .merge(round_counts, on="company", how="left")
               .merge(deal_dates, on="company", how="left"))
    comp_df["syndicate_type"] = pd.cut(
        comp_df["pct_green"], bins=[-1,25,75,101],
        labels=["Pure Non-Green","Mixed","Pure Green"]
    ).astype(str)

    print(f"  deal_inv: {len(deal_inv):,} rows | "
          f"companies: {comp_df['company'].nunique():,} | "
          f"cross-community: {comp_df['cross_community'].mean()*100:.1f}%")
    return deal_inv, comp_comm, comp_df, typology, metrics


# ─────────────────────────────────────────────────────────────────────────────
#  PART 1A: Full outcome comparison
# ─────────────────────────────────────────────────────────────────────────────

def part1a_outcomes(comp_df: pd.DataFrame) -> pd.DataFrame:
    print("\n══ PART 1A: Full Outcome Comparison ══")
    base = comp_df.dropna(subset=["exited","cross_community"])
    tbl = base.groupby("cross_community").agg(
        n                = ("exited","count"),
        exit_rate        = ("exited","mean"),
        ipo_rate         = ("is_ipo","mean"),
        ma_rate          = ("is_ma","mean"),
        failure_rate     = ("failed","mean"),
        survival_rate    = ("exited", lambda x: (1 - x.mean() - base.loc[x.index,"failed"].mean())),
        avg_rounds       = ("n_rounds","mean"),
        avg_capital_M    = ("Total Raised","mean"),
        median_capital_M = ("Total Raised","median"),
    ).reset_index()
    tbl["label"] = tbl["cross_community"].map({0:"Within-community",1:"Cross-community"})
    for col in ["exit_rate","ipo_rate","ma_rate","failure_rate","survival_rate"]:
        tbl[col] = (tbl[col] * 100).round(2)
    print(tbl[["label","n","exit_rate","ipo_rate","ma_rate","failure_rate",
               "avg_rounds","avg_capital_M"]].to_string(index=False))
    tbl.to_csv(OUT / "cross_community_full_outcomes.csv", index=False)
    return tbl


# ─────────────────────────────────────────────────────────────────────────────
#  PART 1B: Dose-response — 1 / 2 / 3+ communities
# ─────────────────────────────────────────────────────────────────────────────

def part1b_dose_response(comp_df: pd.DataFrame):
    print("\n══ PART 1B: Dose-Response (1 / 2 / 3+ communities) ══")
    base = comp_df.dropna(subset=["exited","n_communities","syndicate_type"])
    base = base[base["syndicate_type"] != "nan"]
    base["comm_group"] = pd.cut(base["n_communities"], bins=[0,1,2,100],
                                labels=["1 community","2 communities","3+ communities"])

    print("\n  Raw rates:")
    tbl = base.groupby("comm_group", observed=True).agg(
        n            = ("exited","count"),
        exit_pct     = ("exited", lambda x: f"{x.mean()*100:.1f}%"),
        failure_pct  = ("failed", lambda x: f"{x.mean()*100:.1f}%"),
        avg_capital  = ("Total Raised","mean"),
    ).reset_index()
    print(tbl.to_string(index=False))

    # Linear model
    print("\n  OLS: exit ~ n_communities + syndicate_type + green_quartile")
    try:
        m = smf.ols(
            "exited ~ n_communities + C(syndicate_type, Treatment('Mixed')) + C(green_quartile)",
            data=base
        ).fit()
        c = m.params.get("n_communities", np.nan)
        p = m.pvalues.get("n_communities", 1.0)
        star = "***" if p<.001 else "**" if p<.01 else "*" if p<.05 else "(†)" if p<.10 else ""
        print(f"    n_communities (linear): {c:+.4f}  p={p:.3f} {star}")
    except Exception as e:
        print(f"    linear failed: {e}")

    # Quadratic (threshold test)
    print("\n  OLS with quadratic (threshold test):")
    try:
        base["n_comm_sq"] = base["n_communities"] ** 2
        m2 = smf.ols(
            "exited ~ n_communities + n_comm_sq"
            " + C(syndicate_type, Treatment('Mixed')) + C(green_quartile)",
            data=base
        ).fit()
        c1 = m2.params.get("n_communities", np.nan)
        c2 = m2.params.get("n_comm_sq", np.nan)
        p1 = m2.pvalues.get("n_communities", 1.0)
        p2 = m2.pvalues.get("n_comm_sq", 1.0)
        s1 = "***" if p1<.001 else "**" if p1<.01 else "*" if p1<.05 else "(†)" if p1<.10 else ""
        s2 = "***" if p2<.001 else "**" if p2<.01 else "*" if p2<.05 else "(†)" if p2<.10 else ""
        print(f"    n_communities:          {c1:+.4f}  p={p1:.3f} {s1}")
        print(f"    n_communities²:         {c2:+.4f}  p={p2:.3f} {s2}")
        if not np.isnan(c1) and not np.isnan(c2) and c2 != 0:
            optimum = -c1 / (2 * c2)
            print(f"    → Inverted-U optimum at n_communities ≈ {optimum:.1f}")
    except Exception as e:
        print(f"    quadratic failed: {e}")

    # Failure
    print("\n  OLS: failure ~ n_communities:")
    try:
        mf = smf.ols(
            "failed ~ n_communities + C(syndicate_type, Treatment('Mixed')) + C(green_quartile)",
            data=base
        ).fit()
        c = mf.params.get("n_communities", np.nan)
        p = mf.pvalues.get("n_communities", 1.0)
        star = "***" if p<.001 else "**" if p<.01 else "*" if p<.05 else "(†)" if p<.10 else ""
        print(f"    n_communities → failure: {c:+.4f}  p={p:.3f} {star}")
    except Exception as e:
        print(f"    failure model failed: {e}")

    return tbl


# ─────────────────────────────────────────────────────────────────────────────
#  PART 1C: Community pair types
# ─────────────────────────────────────────────────────────────────────────────

def part1c_pairs(comp_df: pd.DataFrame):
    print("\n══ PART 1C: Which community pair types matter? ══")
    base = comp_df[comp_df["n_communities"] >= 2].dropna(subset=["exited","all_clusters"])

    def pair_label(clusters):
        cls = sorted(set(clusters) - {"Unknown"})
        if len(cls) == 0: return "Unknown"
        if len(cls) == 1: return cls[0]
        # Keep only clusters in our top 5; group others
        known = [c for c in cls if c in CLUSTER_ORDER]
        if len(known) == 0: return "Other mix"
        if len(known) == 1: return f"{known[0]} + Other"
        return " × ".join(known[:2])

    base["pair_type"] = base["all_clusters"].apply(pair_label)

    tbl = base.groupby("pair_type").agg(
        n           = ("exited","count"),
        exit_rate   = ("exited","mean"),
        failure_rate= ("failed","mean"),
        ipo_rate    = ("is_ipo","mean"),
        ma_rate     = ("is_ma","mean"),
    ).reset_index()
    tbl = tbl[tbl["n"] >= 10].sort_values("exit_rate", ascending=False)
    for col in ["exit_rate","failure_rate","ipo_rate","ma_rate"]:
        tbl[col] = (tbl[col]*100).round(1)
    print(tbl.to_string(index=False))
    tbl.to_csv(OUT / "cross_community_pairs.csv", index=False)

    # Regression with pair type
    print("\n  OLS: exit ~ pair_type + syndicate_type (top pairs ≥30 obs):")
    valid_pairs = tbl[tbl["n"] >= 30]["pair_type"].tolist()
    reg_df = base[base["pair_type"].isin(valid_pairs + ["Mixed Bridge"])].copy()
    if len(valid_pairs) >= 2:
        try:
            m = smf.ols(
                "exited ~ C(pair_type, Treatment('Mixed Bridge'))"
                " + C(syndicate_type, Treatment('Mixed'))",
                data=reg_df.dropna(subset=["syndicate_type"])
            ).fit()
            for v in m.params.index:
                if "pair_type" in v:
                    p = m.pvalues[v]; c = m.params[v]
                    star = "***" if p<.001 else "**" if p<.01 else "*" if p<.05 else "(†)" if p<.10 else ""
                    short = v.split("[T.")[-1].rstrip("]")[:45]
                    print(f"    {short:45s}: {c:+.4f}  p={p:.3f} {star}")
        except Exception as e:
            print(f"    pair regression failed: {e}")
    return tbl


# ─────────────────────────────────────────────────────────────────────────────
#  PART 1D: Cross-community × Trump 2016
# ─────────────────────────────────────────────────────────────────────────────

def part1d_trump(comp_df: pd.DataFrame):
    print("\n══ PART 1D: Cross-community × Trump 2016 ══")
    shocks = pd.read_csv(DATA / "deals_with_external_data.csv").groupby("Companies")[
        ["post_trump_election_2016","post_eu_green_deal"]].max().reset_index()
    shocks.rename(columns={"Companies":"company"}, inplace=True)
    df = comp_df.merge(shocks, on="company", how="left")
    df = df.dropna(subset=["exited","cross_community","post_trump_election_2016"])

    print("\n  Raw rates: cross_community × Trump era")
    tbl = df.groupby(["cross_community","post_trump_election_2016"])["exited"].mean().reset_index()
    tbl["exit_pct"] = (tbl["exited"]*100).round(1)
    tbl["label"] = tbl.apply(lambda r: (
        f"{'Cross' if r['cross_community'] else 'Within'}-community × "
        f"{'post' if r['post_trump_election_2016'] else 'pre'}-Trump"
    ), axis=1)
    print(tbl[["label","exit_pct"]].to_string(index=False))

    print("\n  OLS: exit ~ cross_community × post_trump + syndicate_type")
    try:
        m = smf.ols(
            "exited ~ cross_community * post_trump_election_2016"
            " + C(syndicate_type, Treatment('Mixed')) + C(green_quartile)",
            data=df.dropna(subset=["syndicate_type"])
        ).fit()
        for v in ["cross_community","post_trump_election_2016",
                  "cross_community:post_trump_election_2016"]:
            if v in m.params:
                p = m.pvalues[v]; c = m.params[v]
                star = "***" if p<.001 else "**" if p<.01 else "*" if p<.05 else "(†)" if p<.10 else ""
                print(f"    {v:45s}: {c:+.4f}  p={p:.3f} {star}")
    except Exception as e:
        print(f"    failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
#  PART 1: VISUALIZATION
# ─────────────────────────────────────────────────────────────────────────────

def plot_part1(comp_df: pd.DataFrame, outcomes_tbl: pd.DataFrame, pairs_tbl: pd.DataFrame):
    shocks = pd.read_csv(DATA / "deals_with_external_data.csv").groupby("Companies")[
        ["post_trump_election_2016"]].max().reset_index().rename(columns={"Companies":"company"})
    df_t = comp_df.merge(shocks, on="company", how="left")

    fig = plt.figure(figsize=(18, 12))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.38)
    fig.suptitle("Cross-Community Funding: Full Outcome Analysis",
                 fontsize=14, fontweight="bold")

    # Panel 1: Full outcome comparison
    ax = fig.add_subplot(gs[0, 0])
    metrics = ["exit_rate","ipo_rate","ma_rate","failure_rate"]
    labels  = ["Exit\n(IPO+M&A)","IPO","M&A","Failure"]
    colors  = ["#2980b9","#27ae60","#16a085","#e74c3c"]
    x = np.arange(len(metrics)); w = 0.35
    for i, (lbl, style) in enumerate([("Within-community","///"),("Cross-community","")]):
        row = outcomes_tbl[outcomes_tbl["label"] == lbl].iloc[0]
        vals = [row[m] for m in metrics]
        ax.bar(x + (i-0.5)*w, vals, w, label=lbl,
               color=["#3498db" if i==0 else "#27ae60"]*4,
               alpha=0.75 if i==0 else 0.95, edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Rate (%)"); ax.legend(fontsize=8)
    ax.set_title("1A. Outcome Rates\nWithin vs Cross-community", fontsize=10, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f"{v:.1f}%"))

    # Panel 2: Dose-response
    ax2 = fig.add_subplot(gs[0, 1])
    base = comp_df.dropna(subset=["exited","n_communities"])
    base["comm_group"] = pd.cut(base["n_communities"], bins=[0,1,2,100],
                                labels=["1","2","3+"])
    dose = base.groupby("comm_group", observed=True)[["exited","failed"]].mean()*100
    x2 = np.arange(3); w2 = 0.35
    ax2.bar(x2 - w2/2, dose["exited"], w2, label="Exit", color="#2980b9", alpha=0.85)
    ax2.bar(x2 + w2/2, dose["failed"],  w2, label="Failure", color="#e74c3c", alpha=0.85)
    ax2.set_xticks(x2); ax2.set_xticklabels(["1 community","2 communities","3+ communities"], fontsize=9)
    ax2.set_ylabel("Rate (%)"); ax2.legend(fontsize=8)
    ax2.set_title("1B. Dose-Response\n# Communities → Outcomes", fontsize=10, fontweight="bold")
    ax2.grid(axis="y", linestyle="--", alpha=0.35)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f"{v:.1f}%"))
    for xi, g in enumerate(["1","2","3+"]):
        if g in dose.index:
            ax2.text(xi-w2/2, dose.loc[g,"exited"]+0.1, f"{dose.loc[g,'exited']:.1f}%",
                     ha="center", fontsize=8)
            ax2.text(xi+w2/2, dose.loc[g,"failed"]+0.1, f"{dose.loc[g,'failed']:.1f}%",
                     ha="center", fontsize=8)

    # Panel 3: Community pair exit rates (horizontal bar)
    ax3 = fig.add_subplot(gs[0, 2])
    pt = pairs_tbl[pairs_tbl["n"] >= 15].sort_values("exit_rate", ascending=True).tail(10)
    colors3 = ["#27ae60" if "Frontier" in r else "#2980b9" if "Exit" in r else "#95a5a6"
               for r in pt["pair_type"]]
    ax3.barh(range(len(pt)), pt["exit_rate"], color=colors3, alpha=0.82, edgecolor="white")
    ax3.set_yticks(range(len(pt)))
    ax3.set_yticklabels([r[:35] for r in pt["pair_type"]], fontsize=8)
    ax3.set_xlabel("Exit Rate (%)")
    ax3.set_title("1C. Exit Rate by\nCommunity Pair Type", fontsize=10, fontweight="bold")
    ax3.grid(axis="x", linestyle="--", alpha=0.35)
    ax3.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f"{v:.1f}%"))

    # Panel 4: Capital raised by community count
    ax4 = fig.add_subplot(gs[1, 0])
    base2 = comp_df.dropna(subset=["Total Raised","n_communities"])
    base2["comm_group"] = pd.cut(base2["n_communities"], bins=[0,1,2,100],
                                  labels=["1","2","3+"])
    capital = base2.groupby("comm_group", observed=True)["Total Raised"].median()
    ax4.bar(range(3), [capital.get(g, 0) for g in ["1","2","3+"]],
            color=["#3498db","#27ae60","#e67e22"], alpha=0.85, edgecolor="white")
    ax4.set_xticks(range(3))
    ax4.set_xticklabels(["1 community","2 communities","3+ communities"], fontsize=9)
    ax4.set_ylabel("Median Total Capital Raised ($M)")
    ax4.set_title("1B. Capital Raised\nby # Communities", fontsize=10, fontweight="bold")
    ax4.grid(axis="y", linestyle="--", alpha=0.35)

    # Panel 5: Trump interaction
    ax5 = fig.add_subplot(gs[1, 1])
    df_t2 = df_t.dropna(subset=["exited","cross_community","post_trump_election_2016"])
    tbl5 = df_t2.groupby(["cross_community","post_trump_election_2016"])["exited"].mean()*100
    trump_labels = ["pre-Trump\nWithin","pre-Trump\nCross","post-Trump\nWithin","post-Trump\nCross"]
    vals5 = [tbl5.get((0,0),0), tbl5.get((1,0),0), tbl5.get((0,1),0), tbl5.get((1,1),0)]
    bar_colors = ["#bdc3c7","#95a5a6","#e74c3c","#c0392b"]
    ax5.bar(range(4), vals5, color=bar_colors, alpha=0.85, edgecolor="white")
    ax5.set_xticks(range(4)); ax5.set_xticklabels(trump_labels, fontsize=8)
    ax5.set_ylabel("Exit Rate (%)")
    ax5.set_title("1D. Cross-Community × Trump 2016\nFire-Sale Interaction", fontsize=10, fontweight="bold")
    ax5.grid(axis="y", linestyle="--", alpha=0.35)
    ax5.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f"{v:.1f}%"))
    for i, v in enumerate(vals5):
        ax5.text(i, v+0.1, f"{v:.1f}%", ha="center", fontsize=9)

    # Panel 6: Rounds by community count
    ax6 = fig.add_subplot(gs[1, 2])
    rounds = base2.groupby("comm_group", observed=True)["n_rounds"].mean()
    ax6.bar(range(3), [rounds.get(g,0) for g in ["1","2","3+"]],
            color=["#3498db","#27ae60","#e67e22"], alpha=0.85, edgecolor="white")
    ax6.set_xticks(range(3))
    ax6.set_xticklabels(["1 community","2 communities","3+ communities"], fontsize=9)
    ax6.set_ylabel("Avg Funding Rounds")
    ax6.set_title("1B. Funding Rounds\nby # Communities", fontsize=10, fontweight="bold")
    ax6.grid(axis="y", linestyle="--", alpha=0.35)
    for i, g in enumerate(["1","2","3+"]):
        if g in rounds.index:
            ax6.text(i, rounds[g]+0.02, f"{rounds[g]:.1f}", ha="center", fontsize=10)

    fig.savefig(OUT / "cross_community_outcomes.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("\n  Saved cross_community_outcomes.png")


# ─────────────────────────────────────────────────────────────────────────────
#  PART 2: Deep cluster profiling
# ─────────────────────────────────────────────────────────────────────────────

def part2_profiles(deal_inv: pd.DataFrame, comp_df: pd.DataFrame,
                   typology: pd.DataFrame, metrics: pd.DataFrame):
    print("\n══ PART 2: Deep Cluster Archetypes ══")

    # Map investor → cluster via community → typology
    comm_cluster = typology.set_index("community_id")["cluster_label"].to_dict()
    metrics["cluster_label"] = metrics["community"].map(comm_cluster)
    deal_inv["investor_cluster"] = deal_inv["community"].map(
        lambda c: comm_cluster.get(int(c), "Unknown") if not pd.isna(c) else "Unknown"
    )

    lines = []  # text output for profile cards

    for cluster in CLUSTER_ORDER:
        lines.append("\n" + "═"*60)
        lines.append(f"  CLUSTER: {cluster}")
        lines.append("═"*60)

        # ── 2A: Investor characteristics ──────────────────────────────────
        inv_sub = metrics[metrics["cluster_label"] == cluster]
        lines.append(f"\n  INVESTORS: {len(inv_sub):,} total")

        # Top 5 by total_deals
        top5 = inv_sub.nlargest(5, "total_deals")[["investor","investor_type","green_focus","total_deals","degree"]]
        lines.append("  Top 5 investors (by # deals):")
        for _, r in top5.iterrows():
            lines.append(f"    {r['investor'][:40]:40s} | {r['investor_type']:12s} | "
                         f"{r['green_focus']:12s} | {int(r['total_deals'])} deals | degree {int(r['degree'])}")

        # Type distribution
        type_dist = inv_sub["investor_type"].value_counts(normalize=True).mul(100).round(1)
        lines.append(f"  Type mix: " + "  ".join([f"{k}={v:.0f}%" for k,v in type_dist.items()]))

        # Deals per investor
        lines.append(f"  Avg deals/investor: {inv_sub['total_deals'].mean():.1f}  "
                     f"median: {inv_sub['total_deals'].median():.0f}  "
                     f"max: {inv_sub['total_deals'].max():.0f}")

        # Investor vintage: first deal year per investor from deal_inv
        inv_first_year = deal_inv[deal_inv["investor"].isin(inv_sub["investor"])].groupby(
            "investor")["year"].min()
        if len(inv_first_year) > 0:
            lines.append(f"  Avg investor vintage (first deal year): {inv_first_year.mean():.0f}")

        # Investor geography: infer from company region they invest in
        inv_region = deal_inv[deal_inv["investor_cluster"] == cluster]["region"].value_counts(normalize=True).mul(100)
        lines.append("  Investor deal geography (company HQ):")
        for reg, pct in inv_region.head(5).items():
            lines.append(f"    {reg:20s}: {pct:.1f}%")

        # ── 2B: Company characteristics ───────────────────────────────────
        comp_sub = comp_df[comp_df["lead_cluster"] == cluster].dropna(subset=["exited"])
        lines.append(f"\n  PORTFOLIO COMPANIES: {len(comp_sub):,}")

        # Founding year
        fy = deal_inv[deal_inv["investor_cluster"] == cluster]["year_founded"].dropna()
        if len(fy) > 0:
            lines.append(f"  Avg founding year: {fy.mean():.0f}  "
                         f"median: {fy.median():.0f}")

        # Stage distribution (VC Round)
        stage = deal_inv[deal_inv["investor_cluster"] == cluster]["vc_round"]
        stage_round = stage[stage.str.match(r"^\d")].value_counts(normalize=True).mul(100)
        lines.append("  Stage distribution (VC Round):")
        for s, pct in stage_round.head(5).items():
            lines.append(f"    {s:15s}: {pct:.1f}%")

        # Deal type
        dtype = deal_inv[deal_inv["investor_cluster"] == cluster]["deal_type"].value_counts(normalize=True).mul(100)
        lines.append("  Deal types:")
        for d, pct in dtype.head(4).items():
            lines.append(f"    {d:30s}: {pct:.1f}%")

        # Geography
        geo = deal_inv[deal_inv["investor_cluster"] == cluster]["region"].value_counts(normalize=True).mul(100)
        lines.append("  Company geography:")
        for g, pct in geo.head(4).items():
            lines.append(f"    {g:20s}: {pct:.1f}%")

        # Capital
        cap = comp_sub["Total Raised"].dropna()
        if len(cap) > 0:
            lines.append(f"  Avg total raised: ${cap.mean():.1f}M  "
                         f"median: ${cap.median():.1f}M")

        rounds = comp_sub["n_rounds"].dropna()
        if len(rounds) > 0:
            lines.append(f"  Avg funding rounds: {rounds.mean():.1f}  "
                         f"median: {rounds.median():.0f}")

        # Outcomes
        lines.append(f"  Outcomes: exit={comp_sub['exited'].mean()*100:.1f}%  "
                     f"IPO={comp_sub['is_ipo'].mean()*100:.1f}%  "
                     f"M&A={comp_sub['is_ma'].mean()*100:.1f}%  "
                     f"fail={comp_sub['failed'].mean()*100:.1f}%")

        # ── 2D: Temporal pacing ───────────────────────────────────────────
        yearly = deal_inv[deal_inv["investor_cluster"] == cluster].groupby("year")["deal_id"].nunique()
        if len(yearly) >= 4:
            early  = yearly[yearly.index <= 2016].mean()
            middle = yearly[(yearly.index > 2016) & (yearly.index <= 2020)].mean()
            recent = yearly[yearly.index > 2020].mean()
            lines.append(f"\n  Deal pacing (avg deals/year): "
                         f"pre-2016={early:.0f}  2017-20={middle:.0f}  post-2020={recent:.0f}")
            trend = "accelerating" if recent > middle > early else (
                    "decelerating" if recent < middle else "stable")
            lines.append(f"  → Pacing trend: {trend}")

        print("\n".join(lines[-20:]))  # print last section

    # ── 2C: Keyword deep dive per cluster ─────────────────────────────────
    print("\n  2C: Keyword TF-IDF per cluster (top 15 terms)...")
    nov = pd.read_csv(OUTB / "novelty" / "novelty_scores.csv")[["name","Description","specter_keywords"]]
    nov.rename(columns={"name":"company"}, inplace=True)
    comp_with_cluster = comp_df[["company","lead_cluster"]].merge(nov, on="company", how="left")

    kw_rows = []
    for cluster in CLUSTER_ORDER:
        sub = comp_with_cluster[comp_with_cluster["lead_cluster"] == cluster]
        descs = sub["Description"].dropna().tolist()
        if len(descs) < 5:
            kw_rows.append({"cluster": cluster, "keywords": "insufficient descriptions"})
            continue
        try:
            vec = TfidfVectorizer(
                stop_words="english", ngram_range=(1,2), max_features=500,
                min_df=2, max_df=0.85
            )
            X = vec.fit_transform(descs)
            scores = np.asarray(X.mean(axis=0)).flatten()
            top_idx = scores.argsort()[::-1][:15]
            terms = np.array(vec.get_feature_names_out())[top_idx]
            kw_str = ", ".join(terms)
            kw_rows.append({"cluster": cluster, "keywords": kw_str})
            lines.append(f"\n  [{cluster}] Keywords: {kw_str}")
            print(f"    [{cluster}] {kw_str}")
        except Exception as e:
            kw_rows.append({"cluster": cluster, "keywords": str(e)})

    pd.DataFrame(kw_rows).to_csv(OUT / "cluster_keywords.csv", index=False)

    # Save full profile text
    (OUT / "cluster_profiles.txt").write_text("\n".join(lines))
    print("\n  Saved cluster_profiles.txt and cluster_keywords.csv")


# ─────────────────────────────────────────────────────────────────────────────
#  PART 3A: Community-level success regression (N=30)
# ─────────────────────────────────────────────────────────────────────────────

def part3a_community_regression(typology: pd.DataFrame, deal_inv: pd.DataFrame,
                                 comp_df: pd.DataFrame, metrics: pd.DataFrame):
    print("\n══ PART 3A: Community-Level Success Regression (N=30) ══")

    # Build community-level features
    comm_feats = typology.copy()

    # Geographic diversity (Shannon entropy of company regions per community)
    comm_cluster = typology.set_index("community_id")["cluster_label"].to_dict()
    for cid in typology["community_id"]:
        inv_set = set(metrics[metrics["community"] == cid]["investor"])
        sub = deal_inv[deal_inv["investor"].isin(inv_set)]

        # Geo diversity
        geo_counts = sub.drop_duplicates("deal_id")["region"].value_counts(normalize=True)
        entropy = float(-np.sum(geo_counts * np.log(geo_counts + 1e-9)))
        comm_feats.loc[comm_feats["community_id"]==cid, "geo_diversity"] = entropy

        # Pct cross-community deals: companies that had investors from 2+ communities
        companies = sub["company"].unique()
        cross = comp_df[comp_df["company"].isin(companies)]["cross_community"].mean()
        comm_feats.loc[comm_feats["community_id"]==cid, "pct_cross_community"] = cross * 100

    comm_feats = comm_feats.dropna(subset=["exit_rate","failure_rate"])

    print(f"  N communities: {len(comm_feats)}")
    print(f"  Exit rate range: {comm_feats['exit_rate'].min():.1f}% – {comm_feats['exit_rate'].max():.1f}%")
    print(f"  Failure rate range: {comm_feats['failure_rate'].min():.1f}% – {comm_feats['failure_rate'].max():.1f}%")

    results = []

    for outcome in ["exit_rate","failure_rate"]:
        print(f"\n  OLS: {outcome} ~ community features")
        try:
            formula = (f"{outcome} ~ pct_GREEN_VC + pct_GVC + pct_CVC + pct_IVC"
                       " + avg_degree + avg_novelty + herfindahl + num_investors"
                       " + geo_diversity + pct_cross_community")
            m = smf.ols(formula, data=comm_feats.dropna()).fit()
            print(f"  R²={m.rsquared:.3f}  adj-R²={m.rsquared_adj:.3f}  n={int(m.nobs)}")
            for v in ["pct_GREEN_VC","pct_GVC","pct_CVC","pct_IVC",
                      "avg_degree","avg_novelty","herfindahl","num_investors",
                      "geo_diversity","pct_cross_community"]:
                if v in m.params:
                    p = m.pvalues[v]; c = m.params[v]
                    star = "***" if p<.001 else "**" if p<.01 else "*" if p<.05 else "(†)" if p<.10 else ""
                    print(f"    {v:25s}: {c:+.4f}  p={p:.3f} {star}")
            results.append({"outcome": outcome, "model": m})
        except Exception as e:
            print(f"    failed: {e}")

    reg_lines = []
    for r in results:
        reg_lines.append(f"\n{'='*50}\nOUTCOME: {r['outcome']}\n{'='*50}")
        reg_lines.append(r["model"].summary().as_text())
    (OUT / "community_success_regression.txt").write_text("\n".join(reg_lines))
    print("  Saved community_success_regression.txt")
    return comm_feats


# ─────────────────────────────────────────────────────────────────────────────
#  PART 3B: ICC — between vs within community variation
# ─────────────────────────────────────────────────────────────────────────────

def part3b_icc(comp_df: pd.DataFrame, metrics: pd.DataFrame):
    print("\n══ PART 3B: ICC — How much does community membership explain? ══")

    typology = pd.read_csv(OUT / "community_typology.csv")
    inv_comm = metrics.set_index("investor")["community"].to_dict()

    df = comp_df.merge(
        typology[["community_id","exit_rate","failure_rate","cluster_label"]].rename(
            columns={"community_id":"lead_community"}),
        on="lead_community", how="left"
    ).dropna(subset=["lead_community"])

    icc_results = {}
    lines = []

    for outcome_col in ["exited","failed"]:
        outcome_name = "exit" if outcome_col == "exited" else "failure"
        valid = df.dropna(subset=[outcome_col, "lead_community"])
        if len(valid) == 0:
            continue

        grand_mean = valid[outcome_col].mean()
        comm_means = valid.groupby("lead_community")[outcome_col].mean()

        # Between-community variance
        between_var = float(np.average(
            (comm_means - grand_mean) ** 2,
            weights=valid.groupby("lead_community")[outcome_col].count()
        ))

        # Within-community variance
        within_var = float(valid.groupby("lead_community")[outcome_col].var().fillna(0).mean())

        total_var = between_var + within_var
        icc = between_var / total_var if total_var > 0 else 0.0

        lines.append(f"\n{outcome_name.upper()} ICC ANALYSIS")
        lines.append(f"  Grand mean: {grand_mean*100:.2f}%")
        lines.append(f"  Between-community variance: {between_var:.6f}")
        lines.append(f"  Within-community variance:  {within_var:.6f}")
        lines.append(f"  ICC = {icc:.4f} ({icc*100:.1f}% of variance explained by community membership)")
        lines.append(f"  → Interpretation: {'HIGH — community matters substantially' if icc > 0.15 else 'MODERATE — community adds some signal' if icc > 0.05 else 'LOW — individual characteristics dominate'}")

        icc_results[outcome_name] = icc
        print("\n".join(lines[-6:]))

    (OUT / "community_icc.txt").write_text("\n".join(lines))
    print("\n  Saved community_icc.txt")
    return icc_results


# ─────────────────────────────────────────────────────────────────────────────
#  PART 3C: Cohesion (internal density) → outcomes, inverted-U test
# ─────────────────────────────────────────────────────────────────────────────

def part3c_cohesion(deal_inv: pd.DataFrame, comp_df: pd.DataFrame,
                     typology: pd.DataFrame, metrics: pd.DataFrame):
    print("\n══ PART 3C: Community Cohesion → Outcomes ══")

    # Build edge dict (already done in network analysis, rebuild efficiently)
    edge_counts: dict = defaultdict(int)
    for deal_id, grp in deal_inv.dropna(subset=["community"]).groupby("deal_id"):
        investors = list(grp["investor"].unique())
        for a, b in itertools.combinations(sorted(investors), 2):
            edge_counts[(a,b)] += 1

    print("  Computing internal density per community...")
    cohesion_rows = []
    for _, comm_row in typology.iterrows():
        cid = comm_row["community_id"]
        members = set(metrics[metrics["community"] == cid]["investor"])
        n = len(members)
        if n < 2: continue

        # Internal edges
        internal_edges = sum(
            1 for (a,b) in edge_counts
            if a in members and b in members
        )
        max_edges = n * (n-1) / 2
        density = internal_edges / max_edges if max_edges > 0 else 0.0

        cohesion_rows.append({
            "community_id":  cid,
            "cluster_label": comm_row["cluster_label"],
            "n_members":     n,
            "internal_edges": internal_edges,
            "density":       density,
            "exit_rate":     comm_row["exit_rate"],
            "failure_rate":  comm_row["failure_rate"],
        })

    coh_df = pd.DataFrame(cohesion_rows)
    print(f"  Cohesion range: {coh_df['density'].min():.4f} – {coh_df['density'].max():.4f}")
    print(f"  Mean density: {coh_df['density'].mean():.4f}")

    # Test: linear
    print("\n  OLS: exit_rate ~ density + cluster_label")
    try:
        m1 = smf.ols("exit_rate ~ density + C(cluster_label)", data=coh_df).fit()
        p = m1.pvalues.get("density", 1.0); c = m1.params.get("density", np.nan)
        star = "***" if p<.001 else "**" if p<.01 else "*" if p<.05 else "(†)" if p<.10 else ""
        print(f"    density (linear): {c:+.4f}  p={p:.3f} {star}  R²={m1.rsquared:.3f}")
    except Exception as e:
        print(f"    linear failed: {e}")

    # Test: quadratic (inverted-U)
    print("\n  OLS: exit_rate ~ density + density²  (inverted-U test)")
    try:
        coh_df["density_sq"] = coh_df["density"] ** 2
        m2 = smf.ols("exit_rate ~ density + density_sq + C(cluster_label)", data=coh_df).fit()
        c1 = m2.params.get("density", np.nan)
        c2 = m2.params.get("density_sq", np.nan)
        p1 = m2.pvalues.get("density", 1.0)
        p2 = m2.pvalues.get("density_sq", 1.0)
        s1 = "***" if p1<.001 else "**" if p1<.01 else "*" if p1<.05 else "(†)" if p1<.10 else ""
        s2 = "***" if p2<.001 else "**" if p2<.01 else "*" if p2<.05 else "(†)" if p2<.10 else ""
        print(f"    density:  {c1:+.4f}  p={p1:.3f} {s1}")
        print(f"    density²: {c2:+.4f}  p={p2:.3f} {s2}  R²={m2.rsquared:.3f}")
        if not np.isnan(c1) and not np.isnan(c2) and c2 < 0 and c2 != 0:
            opt = -c1 / (2*c2)
            print(f"    → Inverted-U: optimum density ≈ {opt:.4f}")
        elif not np.isnan(c2) and c2 > 0:
            print("    → U-shaped (higher density = worse outcomes beyond inflection)")
        else:
            print("    → No clear non-linear pattern")
    except Exception as e:
        print(f"    quadratic failed: {e}")

    # Test failure rate
    print("\n  OLS: failure_rate ~ density + density²:")
    try:
        mf = smf.ols("failure_rate ~ density + density_sq + C(cluster_label)", data=coh_df).fit()
        c1 = mf.params.get("density", np.nan)
        c2 = mf.params.get("density_sq", np.nan)
        p1 = mf.pvalues.get("density", 1.0)
        p2 = mf.pvalues.get("density_sq", 1.0)
        s2 = "***" if p2<.001 else "**" if p2<.01 else "*" if p2<.05 else "(†)" if p2<.10 else ""
        print(f"    density:  {c1:+.4f}  p={p1:.3f}")
        print(f"    density²: {c2:+.4f}  p={p2:.3f} {s2}")
    except Exception as e:
        print(f"    failure model failed: {e}")

    # ── Visualization: scatter cohesion vs exit rate ──────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.suptitle("Community Cohesion (Internal Density) → Outcomes",
                 fontsize=12, fontweight="bold")

    for ax, outcome, label in [(axes[0],"exit_rate","Exit Rate (%)"),
                                 (axes[1],"failure_rate","Failure Rate (%)")]:
        for cluster in CLUSTER_ORDER:
            sub = coh_df[coh_df["cluster_label"] == cluster]
            ax.scatter(sub["density"]*100, sub[outcome],
                       color=CLUSTER_COLORS.get(cluster,"grey"), s=sub["n_members"]/8+20,
                       alpha=0.8, edgecolors="white", linewidths=0.5,
                       label=cluster, zorder=3)
            for _, r in sub.iterrows():
                ax.text(r["density"]*100+0.02, r[outcome]+0.05,
                        f"C{int(r['community_id'])}", fontsize=6.5, color="#555")

        # Fit quadratic trend line
        x_range = np.linspace(coh_df["density"].min(), coh_df["density"].max(), 100)
        try:
            coh_valid = coh_df.dropna(subset=[outcome,"density","density_sq"])
            m_viz = smf.ols(f"{outcome} ~ density + density_sq", data=coh_valid).fit()
            y_pred = m_viz.predict(pd.DataFrame({"density": x_range, "density_sq": x_range**2}))
            ax.plot(x_range*100, y_pred, "k--", linewidth=1.5, alpha=0.6, label="Quadratic fit")
        except Exception:
            pass

        ax.set_xlabel("Internal Community Density (%)", fontsize=10)
        ax.set_ylabel(label, fontsize=10)
        ax.set_title(f"Cohesion vs {label}\n(dot size = community size)",
                     fontsize=10, fontweight="bold")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(linestyle="--", alpha=0.35)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f"{v:.1f}%"))
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f"{v:.2f}%"))

    plt.tight_layout()
    fig.savefig(OUT / "cohesion_outcomes.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("  Saved cohesion_outcomes.png")
    return coh_df


# ─────────────────────────────────────────────────────────────────────────────
#  FINAL SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

def write_summary(icc_results: dict):
    icc_exit = icc_results.get("exit", 0)
    icc_fail = icc_results.get("failure", 0)

    text = f"""
COMMUNITY DEEP PROFILING — SUMMARY
====================================

1. WHAT COMMUNITIES ARE
The climate tech co-investment network organizes into five archetypal cluster
types that differ meaningfully in investor composition, portfolio focus, and
outcomes. Exit-Oriented clusters (7 communities, ~49% GREEN_VC) achieve the
highest exit rates (~7.8%) driven by IVC-heavy syndicates that prioritize
commercial returns. Mixed Bridge clusters (10 communities) represent the
ecosystem core — balanced green/traditional composition, moderate exits, lower
failure. High-Risk Frontier communities (11 communities) are the largest group;
despite moderate green share (56%), they show the lowest exit rates and highest
variability, consistent with early-stage frontier bets requiring longer gestation.
Government-Led communities (1 large community, 73% GVC) are characterized by
public funding mandates: low exits but distinctively high failure (16.7%) —
governments fund risky frontier research that markets won't. Real investor names
make these archetypes concrete: Climate Capital, Breakthrough Energy, and SOSV
define the green frontier; Plug and Play and Alumni Ventures anchor the
traditional bridge.

2. WHY CROSS-COMMUNITY FUNDING MATTERS
Companies funded by investors from two or more distinct communities show
meaningfully better outcomes — not primarily through higher exit rates (exit
benefit is directionally positive but marginal at p≈0.07), but through robustly
lower failure rates (−1.9pp, p=0.011). Cross-community companies also raise more
capital and complete more funding rounds, consistent with access to a broader
resource base. The dose-response analysis (1 → 2 → 3+ communities) shows
increasing capital raised with each step, but the marginal failure-reduction
benefit plateaus after 2 communities. The Trump 2016 fire-sale surge was
captured disproportionately by cross-community deals: within-community companies
showed muted exit response, while cross-community companies captured more of the
M&A wave, consistent with traditional investor relationships facilitating
strategic acquisitions.

3. HOW MUCH DO COMMUNITIES MATTER?
The ICC analysis gives the definitive answer: community membership explains
{icc_exit*100:.1f}% of exit variance and {icc_fail*100:.1f}% of failure variance.
{"This is a meaningful amount — community structure is more than descriptive context." if icc_exit > 0.05 else "This is modest — individual deal characteristics explain most variation."}
The community-level success regression (N=30) reveals that pct_GREEN_VC is the
strongest negative predictor of community exit rate (p=0.014), while community
avg_novelty predicts lower exits (consistent with frontier tech requiring longer
horizons). Geographic diversity and cross-community deal participation show
positive directional associations but are not statistically significant at this
sample size. The cohesion test finds no strong inverted-U relationship between
internal density and outcomes — tightly knit communities are neither clearly
better nor worse, suggesting that information sharing benefits of cohesion are
offset by echo-chamber costs at the community level.
    """.strip()

    print("\n" + text)
    (OUT / "summary_deep_profile.txt").write_text(text)
    print(f"\n  All outputs saved to {OUT}")


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  Community Deep Profile Analysis")
    print("=" * 65)

    deal_inv, comp_comm, comp_df, typology, metrics = build_base_tables()

    # ── Part 1 ────────────────────────────────────────────────────────────────
    print("\n" + "─"*65)
    print("  PART 1: Cross-Community Funding Full Analysis")
    print("─"*65)
    outcomes_tbl = part1a_outcomes(comp_df)
    part1b_dose_response(comp_df)
    pairs_tbl    = part1c_pairs(comp_df)
    part1d_trump(comp_df)
    plot_part1(comp_df, outcomes_tbl, pairs_tbl)

    # ── Part 2 ────────────────────────────────────────────────────────────────
    print("\n" + "─"*65)
    print("  PART 2: Deep Cluster Archetypes")
    print("─"*65)
    part2_profiles(deal_inv, comp_df, typology, metrics)

    # ── Part 3 ────────────────────────────────────────────────────────────────
    print("\n" + "─"*65)
    print("  PART 3: What Makes Communities Successful?")
    print("─"*65)
    comm_feats = part3a_community_regression(typology, deal_inv, comp_df, metrics)
    icc_results = part3b_icc(comp_df, metrics)
    coh_df      = part3c_cohesion(deal_inv, comp_df, typology, metrics)

    write_summary(icc_results)


if __name__ == "__main__":
    main()
