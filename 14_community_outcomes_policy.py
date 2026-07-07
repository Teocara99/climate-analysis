"""
14 — Community structure × outcomes × policy shocks.

Three focused analyses:
  A. Does community-level structure predict outcomes BEYOND individual syndicate_type?
     Multi-level OLS/logit: exit ~ syndicate_type + community_pct_green +
     community_avg_novelty + community_herfindahl + controls
     → If community vars significant after controlling for individual type:
       community structure adds explanatory power (mechanism worth reporting).

  B. Did policy shocks affect different community types differently?
     exit ~ community_type × post_trump_2016 + community_type × post_eu_green_deal
     (deal-level regression joining deals_with_external_data.csv)

  C. Does cross-community funding predict success?
     Per-company: count distinct investor communities → cross_community dummy
     exit ~ cross_community + syndicate_type + novelty + controls

All models include a plain-language verdict at the end:
  "Include as mechanism" vs "Mention briefly as context".
"""
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import numpy as np
import re
import statsmodels.formula.api as smf
import statsmodels.api as sm
from pathlib import Path
from collections import defaultdict
from load_data import load_deals

OUT  = Path(__file__).parent / "output" / "community"
OUTN = Path(__file__).parent / "output" / "network"
OUTB = Path(__file__).parent / "output"
DATA = Path(__file__).parent / "data"

# ─────────────────────────────────────────────────────────────────────────────
#  SHARED BUILD: company → community mapping
# ─────────────────────────────────────────────────────────────────────────────

def build_company_community_table() -> pd.DataFrame:
    """
    For every company in the dataset build:
      - community_ids: list of distinct communities its investors belong to
      - n_communities: how many distinct communities
      - cross_community: 1 if investors span 2+ communities
      - lead_community_id: most frequent community among its investors
      - community_pct_green: weighted avg pct_GREEN_VC of investor communities
      - community_herfindahl: lead community's Herfindahl
      - community_avg_novelty: lead community's avg_novelty
      - cluster_label: lead community's cluster label
    """
    print("  Building company → community table...")

    deals_raw = load_deals()
    clf = pd.read_csv(OUTB / "investors_classified_output.csv")
    clf = clf[~clf["investor_type"].isin(["PARSE_ERROR"])]
    clf_map = clf.set_index("investor_name")["investor_type"].to_dict()

    metrics  = pd.read_csv(OUTN / "network_metrics.csv")
    inv_comm = metrics.set_index("investor")["community"].to_dict()

    typology = pd.read_csv(OUT / "community_typology.csv")
    comm_features = typology.set_index("community_id")[
        ["pct_GREEN_VC","herfindahl","avg_novelty","cluster_label","cluster"]
    ].to_dict("index")

    def clean_name(raw): return re.sub(r"\([^)]*\)", "", str(raw)).strip()

    company_rows = defaultdict(lambda: {"investors": [], "communities": []})

    for _, row in deals_raw.iterrows():
        if pd.isna(row.get("Investors", None)): continue
        names = [clean_name(n) for n in re.split(r",\s*", str(row["Investors"]).replace("\n", ", "))]
        company = row["Companies"]
        for nm in names:
            if nm and nm in clf_map:
                company_rows[company]["investors"].append(nm)
                comm = inv_comm.get(nm, np.nan)
                if not pd.isna(comm):
                    company_rows[company]["communities"].append(int(comm))

    rows = []
    for company, data in company_rows.items():
        comms = data["communities"]
        if not comms:
            continue
        from collections import Counter
        comm_counter = Counter(comms)
        lead_comm = comm_counter.most_common(1)[0][0]
        distinct_comms = list(set(comms))
        n_comms = len(distinct_comms)

        # Weighted avg pct_GREEN_VC across all communities this company touches
        total_w = sum(comm_counter.values())
        weighted_green = sum(
            comm_features[c]["pct_GREEN_VC"] * w / total_w
            for c, w in comm_counter.items() if c in comm_features
        )
        lead_feat = comm_features.get(lead_comm, {})

        rows.append({
            "Companies":             company,
            "lead_community_id":     lead_comm,
            "n_communities":         n_comms,
            "cross_community":       int(n_comms >= 2),
            "community_pct_green":   weighted_green,
            "community_herfindahl":  lead_feat.get("herfindahl", np.nan),
            "community_avg_novelty": lead_feat.get("avg_novelty", np.nan),
            "cluster_label":         lead_feat.get("cluster_label", "Unknown"),
            "cluster_id":            lead_feat.get("cluster", np.nan),
        })

    df = pd.DataFrame(rows)
    print(f"    {len(df):,} companies mapped to communities")
    return df


def load_company_level() -> pd.DataFrame:
    """
    Merge: company_community + company_investor_mix + novelty + deals (for year).
    Returns a company-level dataframe ready for regression.
    """
    cc = build_company_community_table()

    mix = pd.read_csv(OUTB / "company_investor_mix.csv")
    mix["exited"] = mix["outcome"].isin(["Acquired","IPO / Public"]).astype(int)
    mix["failed"] = (mix["outcome"] == "Failed").astype(int)
    mix["is_ipo"] = (mix["outcome"] == "IPO / Public").astype(int)
    mix["is_ma"]  = (mix["outcome"] == "Acquired").astype(int)

    nov = pd.read_csv(OUTB / "novelty" / "novelty_scores.csv")[["name","specter_novelty"]]
    nov.rename(columns={"name":"Companies","specter_novelty":"novelty"}, inplace=True)

    df = mix.merge(cc, on="Companies", how="inner")
    df = df.merge(nov, on="Companies", how="left")

    # Derive syndicate_type from pct_green (col is in 0-100 scale in mix)
    df["syndicate_type"] = pd.cut(
        df["pct_green"], bins=[-1, 25, 75, 101],
        labels=["Pure Non-Green", "Mixed", "Pure Green"]
    ).astype(str)

    df["novelty_z"] = (df["novelty"] - df["novelty"].mean()) / df["novelty"].std()

    print(f"  Company-level dataset: {len(df):,} companies")
    print(f"  exit rate={df['exited'].mean()*100:.1f}%  "
          f"fail rate={df['failed'].mean()*100:.1f}%  "
          f"cross_community={df['cross_community'].mean()*100:.1f}%")
    return df


def load_deal_level() -> pd.DataFrame:
    """
    Deal-level dataset: deals_with_external_data + community features of each deal.
    Used for Analysis B (policy shock interactions).
    """
    print("  Building deal-level dataset...")
    deals_ext = pd.read_csv(DATA / "deals_with_external_data.csv")
    deals_ext["exited"] = 0  # deal-level: company outcome will be merged

    # Load company outcomes
    mix = pd.read_csv(OUTB / "company_investor_mix.csv")
    mix["exited"] = mix["outcome"].isin(["Acquired","IPO / Public"]).astype(int)
    mix["failed"] = (mix["outcome"] == "Failed").astype(int)
    company_outcome = mix.set_index("Companies")[["exited","failed"]].to_dict("index")

    # Load company → community mapping
    cc = build_company_community_table()
    comm_map = cc.set_index("Companies")[["cluster_label","cluster_id","community_pct_green",
                                          "community_avg_novelty","cross_community"]].to_dict("index")

    deals_raw = load_deals()
    deals_raw["Year"] = pd.to_datetime(deals_raw["Deal Date"], errors="coerce").dt.year

    merged = deals_ext.merge(
        deals_raw[["Deal ID","Companies","Year"]].rename(columns={"Companies":"Companies_raw"}),
        on="Deal ID", how="left"
    )
    merged["Companies"] = merged["Companies_raw"].fillna(merged["Companies"])

    # Add company outcomes
    merged["exited"] = merged["Companies"].map(lambda c: company_outcome.get(c, {}).get("exited", np.nan))
    merged["failed"] = merged["Companies"].map(lambda c: company_outcome.get(c, {}).get("failed", np.nan))

    # Add community features
    merged["cluster_label"] = merged["Companies"].map(
        lambda c: comm_map.get(c, {}).get("cluster_label", np.nan))
    merged["community_pct_green"] = merged["Companies"].map(
        lambda c: comm_map.get(c, {}).get("community_pct_green", np.nan))
    merged["cross_community"] = merged["Companies"].map(
        lambda c: comm_map.get(c, {}).get("cross_community", np.nan))

    print(f"  Deal-level dataset: {len(merged):,} deals")
    return merged


# ─────────────────────────────────────────────────────────────────────────────
#  ANALYSIS A: community vars add explanatory power beyond syndicate_type?
# ─────────────────────────────────────────────────────────────────────────────

def analysis_A(comp_df: pd.DataFrame) -> dict:
    print("\n" + "═"*65)
    print("  ANALYSIS A: Community structure beyond syndicate_type?")
    print("═"*65)

    base = comp_df.dropna(subset=["exited","syndicate_type","community_pct_green",
                                   "community_herfindahl","cluster_label"])
    base = base[base["syndicate_type"] != "nan"]

    results = {}

    # ── Model A1: individual only (baseline) ─────────────────────────────────
    try:
        m1 = smf.ols("exited ~ C(syndicate_type, Treatment('Mixed')) + C(green_quartile)",
                     data=base).fit()
        results["A1_individual_only"] = m1
        print(f"\n  Model A1 — Individual only (baseline)")
        print(f"  R²={m1.rsquared:.4f}  AIC={m1.aic:.1f}  n={int(m1.nobs)}")
        for v in ["C(syndicate_type, Treatment('Mixed'))[T.Pure Green]",
                  "C(syndicate_type, Treatment('Mixed'))[T.Pure Non-Green]"]:
            if v in m1.params:
                p = m1.pvalues[v]; c = m1.params[v]
                star = "***" if p<.001 else "**" if p<.01 else "*" if p<.05 else "(†)" if p<.10 else ""
                print(f"    {v[-25:]:25s}: {c:+.4f}  p={p:.3f} {star}")
    except Exception as e:
        print(f"  A1 failed: {e}")

    # ── Model A2: + community-level vars ────────────────────────────────────
    try:
        m2 = smf.ols(
            "exited ~ C(syndicate_type, Treatment('Mixed')) + C(green_quartile)"
            " + community_pct_green + community_herfindahl + community_avg_novelty",
            data=base.dropna(subset=["community_avg_novelty"])
        ).fit()
        results["A2_plus_community"] = m2
        print(f"\n  Model A2 — + Community variables")
        print(f"  R²={m2.rsquared:.4f}  AIC={m2.aic:.1f}  n={int(m2.nobs)}")
        for v in ["community_pct_green","community_herfindahl","community_avg_novelty"]:
            if v in m2.params:
                p = m2.pvalues[v]; c = m2.params[v]
                star = "***" if p<.001 else "**" if p<.01 else "*" if p<.05 else "(†)" if p<.10 else ""
                print(f"    {v:30s}: {c:+.6f}  p={p:.3f} {star}")
    except Exception as e:
        print(f"  A2 failed: {e}")

    # ── Model A3: + cluster type dummies ────────────────────────────────────
    try:
        m3 = smf.ols(
            "exited ~ C(syndicate_type, Treatment('Mixed')) + C(green_quartile)"
            " + C(cluster_label, Treatment('Mixed Bridge'))",
            data=base
        ).fit()
        results["A3_plus_cluster"] = m3
        print(f"\n  Model A3 — + Cluster type dummies (vs Mixed Bridge baseline)")
        print(f"  R²={m3.rsquared:.4f}  AIC={m3.aic:.1f}  n={int(m3.nobs)}")
        for v in m3.params.index:
            if "cluster_label" in v:
                p = m3.pvalues[v]; c = m3.params[v]
                star = "***" if p<.001 else "**" if p<.01 else "*" if p<.05 else "(†)" if p<.10 else ""
                short = v.split("[T.")[-1].rstrip("]")
                print(f"    {short:25s}: {c:+.4f}  p={p:.3f} {star}")
    except Exception as e:
        print(f"  A3 failed: {e}")

    # ── F-test: do community vars jointly add? ───────────────────────────────
    try:
        m1_n = smf.ols("exited ~ C(syndicate_type, Treatment('Mixed')) + C(green_quartile)",
                       data=base.dropna(subset=["community_avg_novelty"])).fit()
        m2_n = smf.ols(
            "exited ~ C(syndicate_type, Treatment('Mixed')) + C(green_quartile)"
            " + community_pct_green + community_herfindahl + community_avg_novelty",
            data=base.dropna(subset=["community_avg_novelty"])
        ).fit()
        f_stat = ((m1_n.ssr - m2_n.ssr) / 3) / (m2_n.ssr / m2_n.df_resid)
        from scipy import stats
        f_p = stats.f.sf(f_stat, 3, m2_n.df_resid)
        print(f"\n  F-test (community vars jointly, df=3): F={f_stat:.3f}  p={f_p:.4f}")
        results["f_stat"] = f_stat
        results["f_pval"] = f_p
        results["r2_baseline"] = m1_n.rsquared
        results["r2_community"] = m2_n.rsquared
        results["delta_r2"] = m2_n.rsquared - m1_n.rsquared
    except Exception as e:
        print(f"  F-test failed: {e}")

    # ── Verdict ──────────────────────────────────────────────────────────────
    fp  = results.get("f_pval", 1.0)
    dr2 = results.get("delta_r2", 0.0)
    print("\n  ── VERDICT ──")
    if fp < 0.05:
        verdict = (f"INCLUDE AS MECHANISM. Community variables jointly significant "
                   f"(F-test p={fp:.3f}), adding ΔR²={dr2:.4f} beyond syndicate_type. "
                   "Community structure captures co-investment ecosystem dynamics "
                   "not reducible to individual investor green classification.")
    elif fp < 0.15:
        verdict = (f"MARGINAL EVIDENCE (p={fp:.3f}, ΔR²={dr2:.4f}). "
                   "Community variables show suggestive but not conclusive incremental power. "
                   "Mention as contextual finding; do not build causal claims on it.")
    else:
        verdict = (f"MENTION BRIEFLY. Community variables do NOT add explanatory power "
                   f"beyond syndicate_type (F-test p={fp:.3f}, ΔR²={dr2:.4f}). "
                   "Individual investor classification captures the same signal. "
                   "Use community analysis for descriptive narrative only.")
    print(f"  → {verdict}")
    results["verdict"] = verdict
    return results


# ─────────────────────────────────────────────────────────────────────────────
#  ANALYSIS B: Policy shocks × community type
# ─────────────────────────────────────────────────────────────────────────────

def analysis_B(comp_df: pd.DataFrame) -> dict:
    print("\n" + "═"*65)
    print("  ANALYSIS B: Policy shocks × community type")
    print("═"*65)

    # Join company-level data with policy shock dummies via deals
    deals_ext = pd.read_csv(DATA / "deals_with_external_data.csv")
    deals_raw  = load_deals()
    deals_raw["Year"] = pd.to_datetime(deals_raw["Deal Date"], errors="coerce").dt.year

    # Company → first deal year + policy dummies (take max of each dummy per company)
    shock_cols = ["post_trump_election_2016","post_eu_green_deal",
                  "post_biden_inauguration","post_trump_election_2024"]
    shock_cols = [c for c in shock_cols if c in deals_ext.columns]

    company_shocks = deals_ext.groupby("Companies")[shock_cols + ["Year"]].agg(
        {**{c: "max" for c in shock_cols}, "Year": "min"}
    ).reset_index().rename(columns={"Year": "first_deal_year"})

    df = comp_df.merge(company_shocks, on="Companies", how="left")
    df = df.dropna(subset=["exited","cluster_label"] + shock_cols[:2])
    df = df[df["cluster_label"] != "Unknown"]

    # Collapse tiny clusters to avoid near-empty cells
    count = df["cluster_label"].value_counts()
    df["cluster_grp"] = df["cluster_label"].apply(
        lambda x: x if count.get(x, 0) >= 20 else "Other"
    )

    results = {}

    # ── Model B1: Trump 2016 × cluster type ─────────────────────────────────
    print("\n  Model B1 — Trump 2016 × Community Cluster")
    try:
        m = smf.ols(
            "exited ~ C(cluster_grp, Treatment('Mixed Bridge')) * post_trump_election_2016"
            " + C(syndicate_type, Treatment('Mixed'))",
            data=df
        ).fit()
        results["B1_trump"] = m
        print(f"  R²={m.rsquared:.4f}  n={int(m.nobs)}")
        for v in m.params.index:
            if "post_trump" in v or ("cluster" in v and "post" not in v and "[T." in v):
                p = m.pvalues[v]; c = m.params[v]
                star = "***" if p<.001 else "**" if p<.01 else "*" if p<.05 else "(†)" if p<.10 else ""
                short = v.replace("C(cluster_grp, Treatment('Mixed Bridge'))[T.","").rstrip("]")
                short = short.replace("C(syndicate_type, Treatment('Mixed'))[T.","")
                print(f"    {short[:50]:50s}: {c:+.4f}  p={p:.3f} {star}")
    except Exception as e:
        print(f"  B1 failed: {e}")

    # ── Model B2: EU Green Deal × cluster type ───────────────────────────────
    print("\n  Model B2 — EU Green Deal × Community Cluster")
    try:
        m = smf.ols(
            "exited ~ C(cluster_grp, Treatment('Mixed Bridge')) * post_eu_green_deal"
            " + C(syndicate_type, Treatment('Mixed'))",
            data=df
        ).fit()
        results["B2_eu"] = m
        print(f"  R²={m.rsquared:.4f}  n={int(m.nobs)}")
        for v in m.params.index:
            if "eu_green" in v or ("cluster" in v and "eu" not in v and "[T." in v):
                p = m.pvalues[v]; c = m.params[v]
                star = "***" if p<.001 else "**" if p<.01 else "*" if p<.05 else "(†)" if p<.10 else ""
                short = v.replace("C(cluster_grp, Treatment('Mixed Bridge'))[T.","").rstrip("]")
                print(f"    {short[:50]:50s}: {c:+.4f}  p={p:.3f} {star}")
    except Exception as e:
        print(f"  B2 failed: {e}")

    # ── Raw rates table: exit rate pre/post by cluster ────────────────────────
    print("\n  Raw exit rates by cluster × policy era:")
    for shock_col, label in [("post_trump_election_2016","Trump 2016"),
                               ("post_eu_green_deal","EU Green Deal")]:
        if shock_col not in df.columns: continue
        tbl = df.groupby(["cluster_grp", shock_col])["exited"].agg(
            ["mean","count"]
        ).reset_index()
        tbl.columns = ["cluster","post_shock","exit_rate","n"]
        tbl["exit_rate"] = (tbl["exit_rate"] * 100).round(1)
        tbl["post_shock"] = tbl["post_shock"].map({0: f"pre-{label}", 1: f"post-{label}"})
        pivot = tbl.pivot(index="cluster", columns="post_shock", values="exit_rate")
        print(f"\n  {label}")
        print(pivot.to_string())

    # ── Verdict ──────────────────────────────────────────────────────────────
    print("\n  ── VERDICT ──")
    verdict = ("Policy shocks table shows whether Trump 2016 and EU Green Deal "
               "affected community cluster types differently. Inspect interaction "
               "coefficients above: significant interactions (p<0.10) indicate that "
               "the policy effect was heterogeneous across community types — worth "
               "reporting as a moderating mechanism. Non-significant interactions "
               "suggest uniform policy effects regardless of community structure.")
    print(f"  → {verdict}")
    results["verdict"] = verdict
    return results


# ─────────────────────────────────────────────────────────────────────────────
#  ANALYSIS C: Cross-community funding → outcomes
# ─────────────────────────────────────────────────────────────────────────────

def analysis_C(comp_df: pd.DataFrame) -> dict:
    print("\n" + "═"*65)
    print("  ANALYSIS C: Cross-community funding → outcomes")
    print("═"*65)

    base = comp_df.dropna(subset=["exited","failed","cross_community","syndicate_type"])
    base = base[base["syndicate_type"] != "nan"]

    print(f"\n  N companies: {len(base):,}")
    print(f"  Cross-community: {base['cross_community'].sum():,} ({base['cross_community'].mean()*100:.1f}%)")

    # Raw rates
    print("\n  Raw rates by cross_community:")
    tbl = base.groupby("cross_community").agg(
        n=("exited","count"),
        exit_pct=("exited", lambda x: f"{x.mean()*100:.1f}%"),
        fail_pct=("failed", lambda x: f"{x.mean()*100:.1f}%"),
    ).reset_index()
    tbl["label"] = tbl["cross_community"].map({0:"Within-community",1:"Cross-community"})
    print(tbl[["label","n","exit_pct","fail_pct"]].to_string(index=False))

    results = {}

    # ── Model C1: cross_community only ────────────────────────────────────────
    try:
        m1 = smf.ols("exited ~ cross_community", data=base).fit()
        c  = m1.params.get("cross_community", np.nan)
        p  = m1.pvalues.get("cross_community", 1.0)
        star = "***" if p<.001 else "**" if p<.01 else "*" if p<.05 else "(†)" if p<.10 else ""
        print(f"\n  Model C1 — cross_community only:")
        print(f"    cross_community: {c:+.4f}  p={p:.3f} {star}  (R²={m1.rsquared:.4f})")
        results["C1"] = m1
    except Exception as e:
        print(f"  C1 failed: {e}")

    # ── Model C2: + syndicate_type ────────────────────────────────────────────
    try:
        m2 = smf.ols(
            "exited ~ cross_community + C(syndicate_type, Treatment('Mixed')) + C(green_quartile)",
            data=base
        ).fit()
        c  = m2.params.get("cross_community", np.nan)
        p  = m2.pvalues.get("cross_community", 1.0)
        star = "***" if p<.001 else "**" if p<.01 else "*" if p<.05 else "(†)" if p<.10 else ""
        print(f"\n  Model C2 — + syndicate_type + green_quartile controls:")
        print(f"    cross_community: {c:+.4f}  p={p:.3f} {star}  (R²={m2.rsquared:.4f})")
        results["C2"] = m2
    except Exception as e:
        print(f"  C2 failed: {e}")

    # ── Model C3: + novelty (subset with novelty scores) ────────────────────
    nov_base = base.dropna(subset=["novelty_z"])
    try:
        m3 = smf.ols(
            "exited ~ cross_community + C(syndicate_type, Treatment('Mixed'))"
            " + novelty_z + C(green_quartile)",
            data=nov_base
        ).fit()
        c  = m3.params.get("cross_community", np.nan)
        p  = m3.pvalues.get("cross_community", 1.0)
        star = "***" if p<.001 else "**" if p<.01 else "*" if p<.05 else "(†)" if p<.10 else ""
        print(f"\n  Model C3 — + novelty (n={int(m3.nobs)}):")
        print(f"    cross_community: {c:+.4f}  p={p:.3f} {star}  (R²={m3.rsquared:.4f})")
        print(f"    novelty_z:       {m3.params.get('novelty_z',np.nan):+.4f}  "
              f"p={m3.pvalues.get('novelty_z',1):.3f}")
        results["C3"] = m3
    except Exception as e:
        print(f"  C3 failed: {e}")

    # ── Model C4: same for FAILURE ────────────────────────────────────────────
    try:
        m4 = smf.ols(
            "failed ~ cross_community + C(syndicate_type, Treatment('Mixed')) + C(green_quartile)",
            data=base
        ).fit()
        c  = m4.params.get("cross_community", np.nan)
        p  = m4.pvalues.get("cross_community", 1.0)
        star = "***" if p<.001 else "**" if p<.01 else "*" if p<.05 else "(†)" if p<.10 else ""
        print(f"\n  Model C4 — cross_community → FAILURE:")
        print(f"    cross_community: {c:+.4f}  p={p:.3f} {star}  (R²={m4.rsquared:.4f})")
        results["C4_failure"] = m4
    except Exception as e:
        print(f"  C4 failed: {e}")

    # ── n_communities (continuous) ────────────────────────────────────────────
    try:
        m5 = smf.ols(
            "exited ~ n_communities + C(syndicate_type, Treatment('Mixed')) + C(green_quartile)",
            data=base
        ).fit()
        c  = m5.params.get("n_communities", np.nan)
        p  = m5.pvalues.get("n_communities", 1.0)
        star = "***" if p<.001 else "**" if p<.01 else "*" if p<.05 else "(†)" if p<.10 else ""
        print(f"\n  Model C5 — n_communities (continuous) → exit:")
        print(f"    n_communities: {c:+.4f}  p={p:.3f} {star}  (R²={m5.rsquared:.4f})")
        results["C5_continuous"] = m5
    except Exception as e:
        print(f"  C5 failed: {e}")

    # ── Verdict ──────────────────────────────────────────────────────────────
    c2_coef = results.get("C2").params.get("cross_community", np.nan) if "C2" in results else np.nan
    c2_p    = results.get("C2").pvalues.get("cross_community", 1.0)   if "C2" in results else 1.0
    c4_coef = results.get("C4_failure").params.get("cross_community", np.nan) if "C4_failure" in results else np.nan
    c4_p    = results.get("C4_failure").pvalues.get("cross_community", 1.0)   if "C4_failure" in results else 1.0

    print("\n  ── VERDICT ──")
    if c2_p < 0.05:
        verdict = (f"INCLUDE AS MECHANISM. Cross-community funding robustly predicts "
                   f"exits (coef={c2_coef:+.4f}, p={c2_p:.3f}) after controlling for "
                   "syndicate_type. Multi-community syndicates provide resource diversity "
                   "that improves commercial outcomes beyond investor green composition alone.")
    elif c2_p < 0.15:
        verdict = (f"MARGINAL EVIDENCE. Cross-community dummy shows suggestive positive "
                   f"effect on exits (coef={c2_coef:+.4f}, p={c2_p:.3f}). "
                   "Report with caveat: directionally consistent with resource diversity "
                   "hypothesis but below conventional significance threshold.")
    else:
        verdict = (f"MENTION BRIEFLY. Cross-community dummy not significant after "
                   f"controlling for syndicate_type (coef={c2_coef:+.4f}, p={c2_p:.3f}). "
                   "The descriptive rate difference (cross=6.5% vs within=5.8% exit) "
                   "disappears in regression — individual investor classification absorbs "
                   "the cross-community signal. Use raw rates as context only.")
    print(f"  → {verdict}")
    results["verdict"] = verdict
    return results


# ─────────────────────────────────────────────────────────────────────────────
#  VISUALIZATION: 3-panel summary figure
# ─────────────────────────────────────────────────────────────────────────────

def plot_summary(comp_df: pd.DataFrame, results_A: dict, results_C: dict):
    fig, axes = plt.subplots(1, 3, figsize=(17, 6))
    fig.suptitle("Community Structure × Outcomes: Key Results",
                 fontsize=13, fontweight="bold")

    # ── Panel 1: ΔR² decomposition (Analysis A) ───────────────────────────
    ax = axes[0]
    r2_base = results_A.get("r2_baseline", 0)
    r2_comm = results_A.get("r2_community", 0)
    dr2     = results_A.get("delta_r2", 0)
    fp      = results_A.get("f_pval", 1.0)

    bars = ax.bar(["Syndicate\ntype only", "+Community\nvariables"],
                  [r2_base * 100, r2_comm * 100],
                  color=["#3498db","#27ae60" if fp < 0.05 else "#95a5a6"],
                  edgecolor="white", width=0.5, alpha=0.88)
    ax.bar(["Syndicate\ntype only", "+Community\nvariables"],
           [0, dr2 * 100], bottom=[r2_base * 100, r2_base * 100],
           color="#27ae60" if fp < 0.05 else "#95a5a6",
           edgecolor="white", width=0.5, alpha=0.45, hatch="//",
           label=f"ΔR²={dr2*100:.2f}pp (F-test p={fp:.3f})")
    ax.set_ylabel("R² × 100")
    ax.set_title("A. Incremental R²\nDoes community add beyond syndicate_type?",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.2f}%"))
    verdict_short = "✓ Adds value" if fp < 0.05 else ("(†) Marginal" if fp < 0.15 else "✗ No add")
    ax.text(0.5, 0.92, verdict_short, transform=ax.transAxes,
            ha="center", fontsize=11, fontweight="bold",
            color="#27ae60" if fp < 0.05 else ("#e67e22" if fp < 0.15 else "#e74c3c"))

    # ── Panel 2: Exit rate pre/post Trump by cluster (Analysis B) ──────────
    ax = axes[1]
    deals_ext = pd.read_csv(DATA / "deals_with_external_data.csv")
    company_shocks = deals_ext.groupby("Companies")[["post_trump_election_2016"]].max().reset_index()
    df_b = comp_df.merge(company_shocks, on="Companies", how="left")
    df_b = df_b.dropna(subset=["exited","cluster_label","post_trump_election_2016"])
    count = df_b["cluster_label"].value_counts()
    df_b["cluster_grp"] = df_b["cluster_label"].apply(lambda x: x if count.get(x,0) >= 15 else "Other")

    tbl = df_b.groupby(["cluster_grp","post_trump_election_2016"])["exited"].mean().reset_index()
    tbl["exit_pct"] = tbl["exited"] * 100
    tbl["era"] = tbl["post_trump_election_2016"].map({0:"Pre-Trump", 1:"Post-Trump"})
    clusters = sorted(tbl["cluster_grp"].unique())
    x = np.arange(len(clusters))
    w = 0.35
    for i, (era, color) in enumerate([("Pre-Trump","#3498db"),("Post-Trump","#e74c3c")]):
        vals = [tbl[(tbl["cluster_grp"]==c) & (tbl["era"]==era)]["exit_pct"].values[0]
                if len(tbl[(tbl["cluster_grp"]==c) & (tbl["era"]==era)]) > 0 else 0
                for c in clusters]
        ax.bar(x + (i-0.5)*w, vals, w, label=era, color=color, alpha=0.82, edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels([c[:16] for c in clusters], rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Exit Rate (%)")
    ax.set_title("B. Exit Rate Pre/Post Trump 2016\nby Community Cluster Type",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.1f}%"))

    # ── Panel 3: Cross-community exit vs within (Analysis C) ───────────────
    ax = axes[2]
    base = comp_df.dropna(subset=["exited","failed","cross_community"])
    tbl_c = base.groupby("cross_community")[["exited","failed"]].mean() * 100
    x_c = np.array([0, 1])
    w_c = 0.3
    ax.bar(x_c - w_c/2, tbl_c["exited"], w_c, label="Exit Rate", color="#2980b9", alpha=0.85)
    ax.bar(x_c + w_c/2, tbl_c["failed"],  w_c, label="Failure Rate", color="#e74c3c", alpha=0.85)
    ax.set_xticks(x_c)
    ax.set_xticklabels(["Within-community\n(1 community)", "Cross-community\n(2+ communities)"], fontsize=9)
    ax.set_ylabel("Rate (%)")
    ax.set_title("C. Cross-Community Funding\nvs Within-Community",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.1f}%"))
    for xi, col, w_off in [(0,"exited",-w_c/2),(0,"failed",w_c/2),(1,"exited",-w_c/2),(1,"failed",w_c/2)]:
        val = tbl_c[col].iloc[xi]
        ax.text(xi + w_off, val + 0.15, f"{val:.1f}%", ha="center", fontsize=9, fontweight="bold")

    # C verdict
    c2_p = results_C.get("C2").pvalues.get("cross_community", 1.0) if "C2" in results_C else 1.0
    verdict_c = "✓ Significant" if c2_p < 0.05 else ("(†) Marginal" if c2_p < 0.15 else f"✗ p={c2_p:.2f}")
    ax.text(0.5, 0.92, verdict_c, transform=ax.transAxes,
            ha="center", fontsize=11, fontweight="bold",
            color="#27ae60" if c2_p < 0.05 else ("#e67e22" if c2_p < 0.15 else "#e74c3c"))

    plt.tight_layout()
    fig.savefig(OUT / "community_outcomes_policy.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("\n  Saved community_outcomes_policy.png")


# ─────────────────────────────────────────────────────────────────────────────
#  SAVE REGRESSION TABLES
# ─────────────────────────────────────────────────────────────────────────────

def save_tables(results_A, results_B, results_C):
    lines = []

    lines.append("=" * 70)
    lines.append("ANALYSIS A: Community variables beyond syndicate_type")
    lines.append("=" * 70)
    for key in ["A1_individual_only","A2_plus_community","A3_plus_cluster"]:
        if key in results_A:
            m = results_A[key]
            lines.append(f"\n--- {key} ---")
            lines.append(m.summary().as_text())
    lines.append(f"\nF-test community vars jointly: F={results_A.get('f_stat',np.nan):.3f}  "
                 f"p={results_A.get('f_pval',np.nan):.4f}  "
                 f"ΔR²={results_A.get('delta_r2',np.nan):.4f}")
    lines.append(f"VERDICT: {results_A.get('verdict','')}")

    lines.append("\n" + "=" * 70)
    lines.append("ANALYSIS B: Policy shocks × community type")
    lines.append("=" * 70)
    for key in ["B1_trump","B2_eu"]:
        if key in results_B:
            m = results_B[key]
            lines.append(f"\n--- {key} ---")
            lines.append(m.summary().as_text())
    lines.append(f"VERDICT: {results_B.get('verdict','')}")

    lines.append("\n" + "=" * 70)
    lines.append("ANALYSIS C: Cross-community funding → outcomes")
    lines.append("=" * 70)
    for key in ["C1","C2","C3","C4_failure","C5_continuous"]:
        if key in results_C:
            m = results_C[key]
            lines.append(f"\n--- {key} ---")
            lines.append(m.summary().as_text())
    lines.append(f"VERDICT: {results_C.get('verdict','')}")

    (OUT / "community_outcomes_tables.txt").write_text("\n".join(lines))
    print("  Saved community_outcomes_tables.txt")


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  Community Structure × Outcomes × Policy Shocks")
    print("=" * 65)

    comp_df   = load_company_level()
    results_A = analysis_A(comp_df)
    results_B = analysis_B(comp_df)
    results_C = analysis_C(comp_df)
    plot_summary(comp_df, results_A, results_C)
    save_tables(results_A, results_B, results_C)

    print("\n" + "=" * 65)
    print("  SUMMARY OF VERDICTS")
    print("=" * 65)
    print(f"\nA: {results_A.get('verdict','')}")
    print(f"\nB: {results_B.get('verdict','')}")
    print(f"\nC: {results_C.get('verdict','')}")
    print(f"\nAll outputs saved to {OUT}")


if __name__ == "__main__":
    main()
