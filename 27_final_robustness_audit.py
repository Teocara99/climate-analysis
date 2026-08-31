"""
27 — Final Pre-Submission Robustness Audit (9 blocks).
All models at company level (N=6,718).

Outputs → output/robustness_audit/
"""

import warnings; warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd, numpy as np, re, sys, time
from pathlib import Path
from collections import defaultdict
import statsmodels.formula.api as smf
import networkx as nx
import community as community_louvain

sys.path.insert(0, str(Path(__file__).parent))
from load_data import load_deals

ROOT = Path(__file__).parent
OUT  = ROOT / "output" / "robustness_audit"
OUT.mkdir(parents=True, exist_ok=True)

# ── helper ─────────────────────────────────────────────────────────────────────
def sig(p):
    return "***" if p<0.001 else ("**" if p<0.01 else ("*" if p<0.05 else ("†" if p<0.10 else "n.s.")))

def flag(b_base, p_base, b_new, p_new):
    """✅ robust / ⚠️ weakens / ❌ drops."""
    if p_base >= 0.10:             return "–"   # baseline not sig
    if b_new * b_base < 0:         return "❌"  # sign flip
    if p_new >= 0.10:              return "❌"  # drops
    if p_new >= 0.05 and p_base < 0.05: return "⚠️"
    return "✅"

def run_logit(formula, data, key):
    try:
        m = smf.logit(formula, data=data.dropna()).fit(disp=0, maxiter=500)
        b = m.params.get(key, np.nan); p = m.pvalues.get(key, np.nan)
        return b, p, int(m.nobs)
    except:
        return np.nan, np.nan, 0

def run_ols(formula, data, key, cov="HC3"):
    try:
        m = smf.ols(formula, data=data.dropna()).fit(cov_type=cov)
        b = m.params.get(key, np.nan); p = m.pvalues.get(key, np.nan)
        return b, p, int(m.nobs)
    except:
        return np.nan, np.nan, 0

def fmt(b, p, n=None):
    if np.isnan(b): return "n/a"
    s = f"{b:>+.3f}{sig(p)}"
    return s


# ═══════════════════════════════════════════════════════════════════════════════
# LOAD BASE DATA
# ═══════════════════════════════════════════════════════════════════════════════

def load_base():
    print("Loading base data …")
    cdf = pd.read_csv(ROOT/"output/cpu/cpu_company_dataset.csv")
    cim = pd.read_csv(ROOT/"output/company_investor_mix.csv")
    nm  = pd.read_csv(ROOT/"output/network/network_metrics.csv")
    sc  = pd.read_csv(ROOT/"output/bipartite/startup_clusters.csv")
    cp  = pd.read_csv(ROOT/"output/bipartite/cluster_profiles.csv")

    # Merge ESG pct
    cdf = cdf.merge(cim[["Companies","pct_esg","n_investors_matched"]], on="Companies", how="left")
    # Merge tech cluster
    sc["Companies"] = sc["name"]
    cdf = cdf.merge(sc[["Companies","tech_cluster","cluster_name"]], on="Companies", how="left")

    cdf["is_failed"]   = (cdf["outcome"]=="Failed").astype(float)
    cdf["is_exit"]     = cdf["outcome"].isin(["Acquired","IPO / Public"]).astype(float)
    cdf["log_cap"]     = np.log1p(cdf["Total Raised"].fillna(0))
    cdf["nrounds"]     = np.log1p(cdf["n_rounds"].fillna(1))
    cdf["deal_year"]   = cdf["deal_year"].fillna(2018).astype(int)
    cdf["pct_green_sq"]= cdf["pct_green"]**2 / 100   # scale: 0-100 → 0-10000

    # ESG sensitivity versions
    cdf["pct_green_A"] = cdf["pct_green"] + cdf["pct_esg"].fillna(0)  # ESG=GREEN
    cdf["pct_green_B"] = cdf["pct_green"]                              # ESG=TRAD (unchanged green)
    cdf["pct_green_A_sq"] = cdf["pct_green_A"]**2 / 100
    cdf["pct_green_B_sq"] = cdf["pct_green_B"]**2 / 100

    nm["inv_key"] = nm["investor"].str.lower().str.strip()

    print(f"  Companies: {len(cdf):,}  |  sector coverage: {cdf['tech_cluster'].notna().sum():,}")
    print(f"  Cluster dist: {cdf['cluster_name'].value_counts().head(6).to_dict()}")
    return cdf, nm, cp


# ═══════════════════════════════════════════════════════════════════════════════
# BLOCK 1: YEAR FE ON ALL CORE FINDINGS
# ═══════════════════════════════════════════════════════════════════════════════

def block1(cdf, results):
    print("\nBlock 1: Year FE sensitivity …")
    base_ctrl  = "log_cap + nrounds + C(hq_region)"
    fe_ctrl    = base_ctrl + " + C(deal_year)"
    lines = ["\n"+"="*72,
             "BLOCK 1: YEAR FIXED EFFECTS ON ALL CORE FINDINGS",
             "="*72,
             f"  {'Finding':<38}  {'Baseline':>16}  {'+ Year FE':>16}  Flag"]

    specs = [
        ("A. U-shape (green²)", "is_failed",
         f"is_failed ~ pct_green + pct_green_sq + {base_ctrl}",
         f"is_failed ~ pct_green + pct_green_sq + {fe_ctrl}",
         "pct_green_sq"),
        ("B. Pure-green → failure", "is_failed",
         f"is_failed ~ pure_green + {base_ctrl}",
         f"is_failed ~ pure_green + {fe_ctrl}",
         "pure_green"),
        ("C. Pure-trad → exit", "is_exit",
         f"is_exit ~ pure_trad + {base_ctrl}",
         f"is_exit ~ pure_trad + {fe_ctrl}",
         "pure_trad"),
        ("D. Green × novelty", "is_failed",
         f"is_failed ~ pct_green + specter_novelty + pct_green:specter_novelty + {base_ctrl}",
         f"is_failed ~ pct_green + specter_novelty + pct_green:specter_novelty + {fe_ctrl}",
         "pct_green:specter_novelty"),
        ("E. Cross-community → failure", "is_failed",
         f"is_failed ~ cross_community + {base_ctrl}",
         f"is_failed ~ cross_community + {fe_ctrl}",
         "cross_community"),
        ("F. Dose-response (n_communities)", "is_failed",
         f"is_failed ~ n_communities + {base_ctrl}",
         f"is_failed ~ n_communities + {fe_ctrl}",
         "n_communities"),
    ]

    block_results = {}
    for label, dv, formula_base, formula_fe, key in specs:
        b0, p0, n0 = run_logit(formula_base, cdf, key)
        b1, p1, n1 = run_logit(formula_fe,   cdf, key)
        f = flag(b0, p0, b1, p1)
        lines.append(f"  {label:<38}  {fmt(b0,p0):>16}  {fmt(b1,p1):>16}  {f}")
        block_results[label] = {"base":(b0,p0),"fe":(b1,p1),"flag":f,"key":key,"dv":dv}
        results["baseline"][label] = (b0,p0)
        results["fe"][label]       = (b1,p1)
        results["flag_fe"][label]  = f

    return lines, block_results


# ═══════════════════════════════════════════════════════════════════════════════
# BLOCK 2: ESG_ALIGNED SENSITIVITY
# ═══════════════════════════════════════════════════════════════════════════════

def block2(cdf, results):
    print("Block 2: ESG sensitivity …")
    fe_ctrl = "log_cap + nrounds + C(hq_region) + C(deal_year)"
    lines = ["\n"+"="*72,
             "BLOCK 2: ESG_ALIGNED SENSITIVITY",
             f"  ESG investors: 292 of 17,790 (1.6%); company pct_esg mean {cdf['pct_esg'].fillna(0).mean():.2f}%",
             f"  Version A: ESG counted as GREEN (pct_green_A = pct_green + pct_esg)",
             f"  Version B: ESG counted as TRADITIONAL (pct_green unchanged)",
             "="*72,
             f"  {'Finding':<30}  {'Baseline (orig)':>16}  {'ESG=Green':>14}  {'ESG=Trad':>14}"]

    specs = [
        ("A. U-shape (green²)", "is_failed",
         "is_failed ~ pct_green + pct_green_sq",
         "is_failed ~ pct_green_A + pct_green_A_sq",
         "is_failed ~ pct_green_B + pct_green_B_sq",
         "pct_green_sq","pct_green_A_sq","pct_green_B_sq"),
        ("B. Pure-green → failure", "is_failed",
         "is_failed ~ pure_green","is_failed ~ pure_green","is_failed ~ pure_green",
         "pure_green","pure_green","pure_green"),
        ("E. Cross-community", "is_failed",
         "is_failed ~ cross_community","is_failed ~ cross_community","is_failed ~ cross_community",
         "cross_community","cross_community","cross_community"),
    ]

    for label, dv, fb, fa, fbt, kb, ka, kbt in specs:
        ctrl = f" + {fe_ctrl}"
        b0,p0,_ = run_logit(fb+ctrl, cdf, kb)
        ba,pa,_ = run_logit(fa+ctrl, cdf, ka)
        bb,pb,_ = run_logit(fbt+ctrl, cdf, kbt)
        lines.append(f"  {label:<30}  {fmt(b0,p0):>16}  {fmt(ba,pa):>14}  {fmt(bb,pb):>14}")
        results["esg_green"][label] = (ba,pa)
        results["esg_trad"][label]  = (bb,pb)

    # U-shape optimal point under each version
    lines.append("\n  [Optimal green% from U-shape: where d(failure)/d(pct_green) = 0]")
    for label, col_g, col_sq in [
        ("Baseline", "pct_green","pct_green_sq"),
        ("ESG=Green","pct_green_A","pct_green_A_sq"),
    ]:
        sub = cdf.dropna(subset=["is_failed",col_g,col_sq,"log_cap","hq_region"])
        try:
            m = smf.logit(f"is_failed ~ {col_g} + {col_sq} + {fe_ctrl}", data=sub).fit(disp=0,maxiter=500)
            b1 = m.params.get(col_g, np.nan); b2 = m.params.get(col_sq, np.nan)
            # optimal at -b1/(2*b2) where pct_green_sq = pct_green^2/100
            # d/d(g): b1 + b2*(2g/100) = 0 → g = -b1*100/(2*b2)
            if not np.isnan(b2) and b2 != 0:
                opt = -b1 * 100 / (2 * b2)
                lines.append(f"  {label:<14}: optimal pct_green ≈ {opt:.1f}%  "
                             f"(b_green={b1:.4f}, b_sq={b2:.4f})")
        except:
            pass
    return lines


# ═══════════════════════════════════════════════════════════════════════════════
# BLOCK 3: SECTOR ROBUSTNESS
# ═══════════════════════════════════════════════════════════════════════════════

def block3(cdf, results):
    print("Block 3: Sector robustness …")
    fe_ctrl = "log_cap + nrounds + C(hq_region) + C(deal_year)"
    lines = ["\n"+"="*72,
             "BLOCK 3: SECTOR ROBUSTNESS",
             f"  Tech cluster coverage: {cdf['tech_cluster'].notna().sum():,} / {len(cdf):,}",
             "="*72]

    cluster_sizes = cdf["cluster_name"].value_counts()
    lines.append("  Cluster distribution:")
    for cl, n in cluster_sizes.items():
        lines.append(f"    {str(cl):<40}  N={n:,}")

    # 3A: Add sector FE
    lines.append("\n  [3A] + SECTOR (tech_cluster) FIXED EFFECTS")
    sub_sector = cdf.dropna(subset=["tech_cluster"])
    specs = [
        ("A. U-shape",       "is_failed ~ pct_green + pct_green_sq",       "pct_green_sq"),
        ("E. Cross-community","is_failed ~ cross_community",                 "cross_community"),
        ("F. Dose-response",  "is_failed ~ n_communities",                   "n_communities"),
    ]
    for label, f_core, key in specs:
        ctrl_sect = fe_ctrl + " + C(tech_cluster)"
        b_s, p_s, n_s = run_logit(f_core + f" + {ctrl_sect}", sub_sector, key)
        b0, p0 = results["fe"].get(label, (np.nan, np.nan))
        fl = flag(b0, p0, b_s, p_s)
        lines.append(f"  {label:<25}  +sector FE: {fmt(b_s,p_s)}  (baseline+yearFE: {fmt(b0,p0)})  {fl}")
        results["sector_fe"][label] = (b_s, p_s)

    # 3B: Exclude Food/Plant-Based (cluster 4, highest failure)
    food_cluster = cdf["cluster_name"].str.contains("Food|Plant|Meat", na=False)
    food_id = cdf[food_cluster]["tech_cluster"].mode().values[0] if food_cluster.any() else None
    n_food  = food_cluster.sum()
    lines.append(f"\n  [3B] EXCLUDE Food/Plant-Based cluster (N={n_food:,})")
    cdf_nofood = cdf[~food_cluster].copy()
    for label, f_core, key in specs:
        b_nf, p_nf, n_nf = run_logit(f_core + f" + {fe_ctrl}", cdf_nofood, key)
        b0, p0 = results["fe"].get(label, (np.nan, np.nan))
        fl = flag(b0, p0, b_nf, p_nf)
        lines.append(f"  {label:<25}  -food: {fmt(b_nf,p_nf)}  N={n_nf:,}  {fl}")
        results["no_food"][label] = (b_nf, p_nf)

    # 3C: Exclude the 2 largest clusters
    top2_clusters = cluster_sizes.index[:2]
    cdf_notop2 = cdf[~cdf["cluster_name"].isin(top2_clusters)].copy()
    n_excl = cdf[cdf["cluster_name"].isin(top2_clusters)].shape[0]
    lines.append(f"\n  [3C] EXCLUDE top-2 sectors ({', '.join([str(c) for c in top2_clusters])}, N={n_excl:,})")
    for label, f_core, key in specs:
        b_nt, p_nt, n_nt = run_logit(f_core + f" + {fe_ctrl}", cdf_notop2, key)
        b0, p0 = results["fe"].get(label, (np.nan, np.nan))
        fl = flag(b0, p0, b_nt, p_nt)
        lines.append(f"  {label:<25}  -top2: {fmt(b_nt,p_nt)}  N={n_nt:,}  {fl}")
        results["no_top2"][label] = (b_nt, p_nt)

    return lines


# ═══════════════════════════════════════════════════════════════════════════════
# BLOCK 4: REGIONAL SUBSAMPLE
# ═══════════════════════════════════════════════════════════════════════════════

def block4(cdf, results):
    print("Block 4: Regional subsamples …")
    ctrl = "log_cap + nrounds + C(deal_year)"
    lines = ["\n"+"="*72,
             "BLOCK 4: REGIONAL SUBSAMPLE",
             "  Does the U-shape optimum hold within each region?",
             "="*72,
             f"  {'Region':<14}  {'N':>6}  {'green β':>10}  {'green² β':>10}  {'Optimum%':>10}  {'E.CrossComm':>12}"]

    for region in ["Americas","Europe","Asia"]:
        sub = cdf[cdf["hq_region"]==region].copy()
        n = len(sub)
        # U-shape
        b1,p1,_ = run_logit(f"is_failed ~ pct_green + pct_green_sq + {ctrl}", sub, "pct_green")
        b2,p2,_ = run_logit(f"is_failed ~ pct_green + pct_green_sq + {ctrl}", sub, "pct_green_sq")
        opt_str = "n/a"
        if not np.isnan(b2) and b2 != 0 and not np.isnan(b1):
            opt = -b1 * 100 / (2 * b2)
            opt_str = f"{opt:.0f}%" if 0 < opt < 150 else "out of range"
        # Cross-community
        bcc, pcc, _ = run_logit(f"is_failed ~ cross_community + {ctrl}", sub, "cross_community")

        results[f"region_{region}"]["A. U-shape"] = (b2, p2)
        results[f"region_{region}"]["E. Cross-community"] = (bcc, pcc)

        lines.append(f"  {region:<14}  {n:>6,}  {fmt(b1,p1):>10}  {fmt(b2,p2):>10}  "
                     f"{opt_str:>10}  {fmt(bcc,pcc):>12}")

    return lines


# ═══════════════════════════════════════════════════════════════════════════════
# BLOCK 5: COMMUNITY DETECTION ROBUSTNESS
# ═══════════════════════════════════════════════════════════════════════════════

def block5(cdf, nm, results):
    print("Block 5: Community detection robustness …")
    lines = ["\n"+"="*72,
             "BLOCK 5: COMMUNITY DETECTION ROBUSTNESS",
             "  Rebuild co-investment graph, re-run Louvain (res=0.5, 1.0, 2.0) + Leiden",
             "  For each: recompute cross_community per company, re-run Model E",
             "="*72]

    deals = load_deals()
    deals["Deal Date"] = pd.to_datetime(deals["Announced Date"], errors="coerce")
    deals = deals.dropna(subset=["Deal Date","Investors"])

    def parse(s):
        return [n.strip() for n in re.split(r",\s*|\n", str(s)) if n.strip()]

    # Build co-investment graph (investors as nodes, edges = co-invested)
    print("  Building co-investment graph …")
    G = nx.Graph()
    known_investors = set(nm["inv_key"])
    company_investor_map = defaultdict(set)

    for _, row in deals.iterrows():
        invs = [n.lower() for n in parse(row["Investors"]) if n.lower() in known_investors]
        comp = str(row.get("Companies","")).lower().strip()
        for n in invs:
            company_investor_map[comp].add(n)
        for i in range(len(invs)):
            for j in range(i+1, len(invs)):
                if G.has_edge(invs[i], invs[j]):
                    G[invs[i]][invs[j]]["weight"] += 1
                else:
                    G.add_edge(invs[i], invs[j], weight=1)

    G_undirected = nx.Graph(G)
    print(f"  Graph: {G_undirected.number_of_nodes():,} nodes  {G_undirected.number_of_edges():,} edges")

    def compute_cross_community(partition_dict):
        """Given investor→community dict, compute cross_community per company."""
        records = []
        for comp, inv_set in company_investor_map.items():
            comms = {partition_dict.get(inv) for inv in inv_set if inv in partition_dict}
            comms.discard(None)
            records.append({"company_key": comp, "cross_comm_new": int(len(comms) >= 2),
                            "n_comms_new": len(comms)})
        return pd.DataFrame(records)

    cdf["company_key"] = cdf["Companies"].str.lower().str.strip()
    ctrl = "log_cap + nrounds + C(hq_region) + C(deal_year)"

    lines.append(f"\n  {'Method':<20}  {'N communities':>14}  {'cross_comm %':>13}  "
                 f"{'β failure':>12}  {'p':>8}  flag")

    for algo, res_label in [
        ("louvain_05",  "Louvain res=0.5"),
        ("louvain_10",  "Louvain res=1.0 (original)"),
        ("louvain_20",  "Louvain res=2.0"),
        ("leiden",      "Leiden"),
    ]:
        try:
            if algo == "leiden":
                import leidenalg, igraph as ig
                node_list = list(G_undirected.nodes())
                node_idx  = {n:i for i,n in enumerate(node_list)}
                ig_G = ig.Graph(n=len(node_list), directed=False)
                edges_ig  = [(node_idx[u], node_idx[v]) for u,v in G_undirected.edges()]
                weights_ig = [G_undirected[u][v].get("weight",1) for u,v in G_undirected.edges()]
                ig_G.add_edges(edges_ig)
                ig_G.es["weight"] = weights_ig
                partition = leidenalg.find_partition(ig_G, leidenalg.ModularityVertexPartition,
                                                     weights="weight", seed=42)
                part_dict = {node_list[i]: partition.membership[i] for i in range(len(node_list))}
                res_val = 1.0
            else:
                res_map = {"louvain_05":0.5,"louvain_10":1.0,"louvain_20":2.0}
                part_dict = community_louvain.best_partition(G_undirected,
                                                             weight="weight",
                                                             resolution=res_map[algo],
                                                             random_state=42)
                res_val = res_map[algo]

            n_comms = len(set(part_dict.values()))
            cross_df = compute_cross_community(part_dict)
            tmp = cdf.merge(cross_df, on="company_key", how="left")
            cc_pct = tmp["cross_comm_new"].mean()

            b, p, n = run_logit(f"is_failed ~ cross_comm_new + {ctrl}", tmp, "cross_comm_new")
            b0, p0 = results["fe"].get("E. Cross-community",(np.nan,np.nan))
            fl = flag(b0, p0, b, p)
            lines.append(f"  {res_label:<20}  {n_comms:>14,}  {cc_pct:>13.1%}  "
                         f"{fmt(b,p):>12}  {p:>8.4f}  {fl}")
            results[algo]["E. Cross-community"] = (b, p)

        except Exception as e:
            lines.append(f"  {res_label:<20}  ERROR: {e}")
            results[algo]["E. Cross-community"] = (np.nan, np.nan)

    return lines


# ═══════════════════════════════════════════════════════════════════════════════
# BLOCK 6: DOSE-RESPONSE LINEARITY CHECK
# ═══════════════════════════════════════════════════════════════════════════════

def block6(cdf, results):
    print("Block 6: Dose-response linearity …")
    ctrl = "log_cap + nrounds + C(hq_region) + C(deal_year)"
    cdf["n_comms_sq"] = cdf["n_communities"]**2
    lines = ["\n"+"="*72,
             "BLOCK 6: DOSE-RESPONSE LINEARITY CHECK",
             "  H: If n_communities² is significant → threshold effect, not linear",
             "="*72]

    sub = cdf.dropna(subset=["is_failed","n_communities","n_comms_sq","log_cap","hq_region"])
    for formula, key, label in [
        (f"is_failed ~ n_communities + {ctrl}",           "n_communities","Linear only"),
        (f"is_failed ~ n_communities + n_comms_sq + {ctrl}","n_communities","Quadratic — linear term"),
        (f"is_failed ~ n_communities + n_comms_sq + {ctrl}","n_comms_sq",  "Quadratic — squared term"),
    ]:
        b,p,n = run_logit(formula, sub, key)
        lines.append(f"  {label:<35}  β={b:>+.5f}  p={p:.4f} {sig(p)}  N={n:,}")

    # OLS for interpretability
    lines.append("\n  [OLS for marginal effects — linear probability model]")
    m_lin = smf.ols(f"is_failed ~ n_communities + {ctrl}", data=sub).fit(cov_type="HC3")
    m_quad = smf.ols(f"is_failed ~ n_communities + n_comms_sq + {ctrl}", data=sub).fit(cov_type="HC3")
    b_l = m_lin.params.get("n_communities",np.nan); p_l = m_lin.pvalues.get("n_communities",np.nan)
    b_q = m_quad.params.get("n_communities",np.nan); p_q = m_quad.pvalues.get("n_communities",np.nan)
    b_sq= m_quad.params.get("n_comms_sq",np.nan);   p_sq= m_quad.pvalues.get("n_comms_sq",np.nan)
    lines.append(f"  Linear:    n_communities β={b_l:>+.5f}  p={p_l:.4f} {sig(p_l)}")
    lines.append(f"  Quadratic: n_communities β={b_q:>+.5f}  p={p_q:.4f} {sig(p_q)}  "
                 f"n²: β={b_sq:>+.5f}  p={p_sq:.4f} {sig(p_sq)}")
    lines.append(f"  adj-R² linear={m_lin.rsquared_adj:.4f}  quadratic={m_quad.rsquared_adj:.4f}")

    verdict = ("THRESHOLD EFFECT confirmed (n² significant)" if p_sq < 0.05
               else ("marginal non-linearity" if p_sq < 0.10 else "LINEAR confirmed"))
    lines.append(f"  VERDICT: {verdict}")
    results["nonlinear"]["F. Dose-response"] = (b_sq, p_sq)
    return lines


# ═══════════════════════════════════════════════════════════════════════════════
# BLOCK 7: INVESTOR MISCLASSIFICATION SIMULATION (100 runs)
# ═══════════════════════════════════════════════════════════════════════════════

def block7(cdf, nm, results):
    print("Block 7: Misclassification simulation (100 runs) …")
    ctrl = "log_cap + nrounds + C(hq_region) + C(deal_year)"
    lines = ["\n"+"="*72,
             "BLOCK 7: INVESTOR CLASSIFICATION ROBUSTNESS (misclassification simulation)",
             "  Assumption: LLM investor classification is ~85% accurate",
             "  Simulation: randomly flip 15% of investor green/traditional labels per run",
             "  Method: for each company, sample binomial noise on n_green investors",
             "  Repeat 100× → report: mean β, % runs p<0.05 for green² (U-shape)",
             "="*72]

    # Per-company approximation of n_green and n_trad
    # Use pct_green and n_investors_matched to get counts
    cdf2 = cdf.copy()
    cdf2["n_inv_est"] = cdf2["n_investors_matched"].fillna(3).clip(lower=1)
    cdf2["n_green_est"]= (cdf2["pct_green"]/100 * cdf2["n_inv_est"]).round().astype(int)
    cdf2["n_trad_est"] = (cdf2["pct_traditional"]/100 * cdf2["n_inv_est"]).round().astype(int)

    def one_run(seed):
        rng = np.random.default_rng(seed)
        df = cdf2.copy()
        n_inv = df["n_inv_est"].values.astype(int)
        n_g   = df["n_green_est"].values.clip(0)
        n_t   = df["n_trad_est"].values.clip(0)

        # Binomial flips: some greens become trad, some trads become green
        flip_gt = rng.binomial(n_g, 0.15)  # green→trad
        flip_tg = rng.binomial(n_t, 0.15)  # trad→green
        n_g_new = (n_g - flip_gt + flip_tg).clip(0, n_inv)
        df["pct_green_sim"]    = np.where(n_inv>0, n_g_new/n_inv*100, df["pct_green"])
        df["pct_green_sim_sq"] = df["pct_green_sim"]**2 / 100

        try:
            sub = df.dropna(subset=["is_failed","pct_green_sim","pct_green_sim_sq","log_cap","hq_region"])
            m = smf.logit(f"is_failed ~ pct_green_sim + pct_green_sim_sq + {ctrl}",
                          data=sub).fit(disp=0, maxiter=300)
            b = m.params.get("pct_green_sim_sq", np.nan)
            p = m.pvalues.get("pct_green_sim_sq", np.nan)
            return b, p
        except:
            return np.nan, np.nan

    sim_results = [one_run(seed) for seed in range(100)]
    betas = np.array([r[0] for r in sim_results if not np.isnan(r[0])])
    pvals = np.array([r[1] for r in sim_results if not np.isnan(r[1])])
    pct_sig = (pvals < 0.05).mean() * 100
    pct_right_sign = (betas < 0).mean() * 100  # expect negative for U-shape (higher sq → more failure at extremes)

    lines.append(f"\n  N runs completed: {len(betas)} / 100")
    lines.append(f"  U-shape (green²) under 15% misclassification:")
    lines.append(f"    Mean β:         {np.mean(betas):>+.5f}  (baseline: {results['fe'].get('A. U-shape',(np.nan,np.nan))[0]:>+.5f})")
    lines.append(f"    SD β:           {np.std(betas):>.5f}")
    lines.append(f"    % runs p<0.05:  {pct_sig:.1f}%")
    lines.append(f"    % runs correct sign (β<0): {pct_right_sign:.1f}%")
    lines.append(f"    95% CI of β:    [{np.percentile(betas,2.5):>+.5f}, {np.percentile(betas,97.5):>+.5f}]")

    verdict = ("ROBUST to 15% misclassification" if pct_sig >= 80
               else ("PARTIALLY ROBUST" if pct_sig >= 50 else "FRAGILE to misclassification"))
    lines.append(f"  VERDICT: {verdict} (significant in {pct_sig:.0f}% of runs)")
    results["misclass"]["A. U-shape"] = (np.mean(betas), 1-pct_sig/100)  # approx p
    return lines


# ═══════════════════════════════════════════════════════════════════════════════
# BLOCK 8: MEGA-INVESTOR INFLUENCE
# ═══════════════════════════════════════════════════════════════════════════════

def block8(cdf, nm, results):
    print("Block 8: Mega-investor influence …")
    ctrl = "log_cap + nrounds + C(hq_region) + C(deal_year)"
    lines = ["\n"+"="*72,
             "BLOCK 8: MEGA-INVESTOR INFLUENCE",
             "  Drop all companies whose first-round investors include the top-20 by deal count",
             "="*72]

    top20 = nm.nlargest(20, "total_deals")[["investor","total_deals"]].reset_index(drop=True)
    lines.append("  Top-20 investors by deal count:")
    for _, r in top20.iterrows():
        lines.append(f"    {r['investor']:<40}  {int(r['total_deals']):,} deals")

    top20_set = set(top20["investor"].str.lower().str.strip())

    deals = load_deals()
    deals["Deal Date"] = pd.to_datetime(deals["Announced Date"], errors="coerce")
    deals = deals.dropna(subset=["Deal Date","Investors"])
    deals = deals.sort_values("Deal Date")
    deals["company_key"] = deals["Companies"].str.lower().str.strip()
    first_deals = deals.drop_duplicates("company_key", keep="first")

    def has_mega(inv_str):
        names = [n.strip().lower() for n in re.split(r",\s*|\n", str(inv_str)) if n.strip()]
        return any(n in top20_set for n in names)

    first_deals_flags = first_deals.set_index("company_key")["Investors"].apply(has_mega)
    cdf["company_key"] = cdf["Companies"].str.lower().str.strip()
    cdf = cdf.merge(first_deals_flags.rename("has_mega"), on="company_key", how="left")
    cdf_nomega = cdf[~cdf["has_mega"].fillna(False)].copy()

    n_excl = cdf["has_mega"].fillna(False).sum()
    lines.append(f"\n  Companies involving top-20 investors: {n_excl:,} excluded")
    lines.append(f"  Remaining: {len(cdf_nomega):,}")

    specs = [
        ("A. U-shape",        "is_failed ~ pct_green + pct_green_sq", "pct_green_sq"),
        ("E. Cross-community","is_failed ~ cross_community",           "cross_community"),
        ("F. Dose-response",  "is_failed ~ n_communities",             "n_communities"),
    ]
    lines.append(f"\n  {'Finding':<25}  {'Full sample':>16}  {'-MegaInvestors':>16}  flag")
    for label, f_core, key in specs:
        b_full, p_full = results["fe"].get(label, (np.nan, np.nan))
        b_nm, p_nm, n_nm = run_logit(f_core + f" + {ctrl}", cdf_nomega, key)
        fl = flag(b_full, p_full, b_nm, p_nm)
        lines.append(f"  {label:<25}  {fmt(b_full,p_full):>16}  {fmt(b_nm,p_nm):>16}  {fl}")
        results["no_mega"][label] = (b_nm, p_nm)

    return lines, cdf


# ═══════════════════════════════════════════════════════════════════════════════
# BLOCK 9: ALTERNATIVE OUTCOME DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════════

def block9(cdf, results):
    print("Block 9: Alternative outcome definitions …")
    ctrl = "log_cap + nrounds + C(hq_region) + C(deal_year)"
    lines = ["\n"+"="*72,
             "BLOCK 9: ALTERNATIVE OUTCOME DEFINITIONS",
             "="*72]

    # 9A: Multinomial logit (fail vs exit vs active)
    lines.append("\n  [9A] MULTINOMIAL LOGIT: outcome = {fail, exit, active}")
    lines.append("  Baseline = Active/Private; coefficients show relative log-odds")
    from statsmodels.discrete.discrete_model import MNLogit
    outcome_cat = cdf["outcome"].map({
        "Failed":0, "Acquired":1, "IPO / Public":1, "Active / Private":2
    }).fillna(2).astype(int)
    cdf["outcome_3"] = outcome_cat  # 0=fail, 1=exit, 2=active (reference)

    # For multinomial, use OLS proxy (binary outcomes)
    lines.append("  [Logit approximation: separate fail and exit models]")
    for label, f_core, key in [
        ("A. green²→fail",   "is_failed ~ pct_green + pct_green_sq", "pct_green_sq"),
        ("C. pure_trad→exit","is_exit   ~ pure_trad",                "pure_trad"),
        ("E. cross→fail",    "is_failed ~ cross_community",           "cross_community"),
    ]:
        b,p,n = run_logit(f_core + f" + {ctrl}", cdf, key)
        lines.append(f"  {label:<25}  β={b:>+.4f}  p={p:.4f} {sig(p)}  N={n:,}")

    # 9B: Cox proportional hazards
    lines.append("\n  [9B] COX SURVIVAL ANALYSIS")
    try:
        from lifelines import CoxPHFitter
        cdf2 = cdf.copy()
        cdf2["duration"] = 2026 - cdf2["deal_year"].clip(lower=2000, upper=2025)
        cdf2["event"]    = cdf2["is_failed"]
        cdf2 = cdf2.dropna(subset=["duration","event","cross_community","pct_green",
                                    "log_cap","nrounds"])
        cox_cols = ["duration","event","cross_community","pct_green","pct_green_sq",
                    "pure_green","pure_trad","log_cap","nrounds"]
        cph = CoxPHFitter(penalizer=0.1)
        cph.fit(cdf2[cox_cols], duration_col="duration", event_col="event")
        for var in ["cross_community","pct_green","pct_green_sq","pure_green","pure_trad"]:
            if var in cph.params_.index:
                b = cph.params_[var]; p = cph.summary["p"][var]
                lines.append(f"  Cox HR({var}): exp(β)={np.exp(b):.3f}  β={b:>+.4f}  p={p:.4f} {sig(p)}")
        results["cox"]["E. Cross-community"] = (cph.params_.get("cross_community",np.nan),
                                                 cph.summary["p"].get("cross_community",np.nan))
        results["cox"]["A. U-shape"] = (cph.params_.get("pct_green_sq",np.nan),
                                         cph.summary["p"].get("pct_green_sq",np.nan))
    except ImportError:
        lines.append("  lifelines not installed — skipping Cox")
        # Fallback: OLS survival proxy
        cdf2 = cdf.copy()
        cdf2["duration"] = (2026 - cdf2["deal_year"].clip(lower=2000,upper=2025)).astype(float)
        b_surv,p_surv,_ = run_ols(f"duration ~ cross_community + {ctrl}", cdf2, "cross_community")
        lines.append(f"  OLS proxy: duration ~ cross_community β={b_surv:>+.4f} {sig(p_surv)}")
    except Exception as e:
        lines.append(f"  Cox error: {e}")

    # 9C: LPM (linear probability) vs logit comparison
    lines.append("\n  [9C] LINEAR PROBABILITY MODEL vs LOGIT")
    for label, f_core, key in [
        ("E. Cross-community","is_failed ~ cross_community","cross_community"),
        ("F. Dose-response",  "is_failed ~ n_communities",  "n_communities"),
    ]:
        b_log,p_log,n = run_logit(f_core + f" + {ctrl}", cdf, key)
        b_lpm,p_lpm,_ = run_ols(f_core + f" + {ctrl}", cdf, key)
        lines.append(f"  {label:<25}  Logit: β={b_log:>+.4f} {sig(p_log)}  |  "
                     f"LPM: β={b_lpm:>+.4f} {sig(p_lpm)}  N={n:,}")

    return lines


# ═══════════════════════════════════════════════════════════════════════════════
# MASTER ROBUSTNESS TABLE
# ═══════════════════════════════════════════════════════════════════════════════

def master_table(results):
    findings = ["A. U-shape", "B. Pure-green → failure", "C. Pure-trad → exit",
                "D. Green × novelty", "E. Cross-community", "F. Dose-response"]

    cols = [
        ("Baseline",    "baseline"),
        ("+Year FE",    "fe"),
        ("+Sector FE",  "sector_fe"),
        ("-Food",       "no_food"),
        ("-Top2 sector","no_top2"),
        ("Americas",    "region_Americas"),
        ("Europe",      "region_Europe"),
        ("Asia",        "region_Asia"),
        ("ESG=Green",   "esg_green"),
        ("ESG=Trad",    "esg_trad"),
        ("Leiden",      "leiden"),
        ("Lvn res=0.5", "louvain_05"),
        ("Lvn res=2.0", "louvain_20"),
        ("-MegaInv",    "no_mega"),
        ("Misclass85%", "misclass"),
    ]

    # Header
    col_w = 13
    header = f"  {'Finding':<30}" + "".join(f"  {c[0][:col_w]:>{col_w}}" for c in cols)
    sep    = "  " + "-"*30 + "  " + "  ".join("-"*col_w for _ in cols)

    lines = ["\n"+"="*100,
             "MASTER ROBUSTNESS TABLE",
             "  Each cell: β (p-value) or flag [✅ robust  ⚠️ weakens  ❌ drops  – not sig baseline]",
             "="*100, header, sep]

    for finding in findings:
        b0, p0 = results["baseline"].get(finding, (np.nan, np.nan))
        row = f"  {finding:<30}"
        for col_label, col_key in cols:
            b, p = results.get(col_key, {}).get(finding, (np.nan, np.nan))
            if np.isnan(b):
                cell = "   n/a"
            else:
                fl = flag(b0, p0, b, p)
                cell = f"{fl} {fmt(b,p)}"
            row += f"  {cell:>{col_w}}"
        lines.append(row)

    # Flag summary
    lines += ["", "  FINDINGS THAT DROPPED (❌) IN ANY CHECK:"]
    for finding in findings:
        b0, p0 = results["baseline"].get(finding, (np.nan, np.nan))
        if p0 >= 0.10: continue
        drops = []
        for col_label, col_key in cols:
            b, p = results.get(col_key, {}).get(finding, (np.nan, np.nan))
            if flag(b0, p0, b, p) == "❌":
                drops.append(col_label)
        if drops:
            lines.append(f"    {finding}: ❌ in {', '.join(drops)}")
        else:
            lines.append(f"    {finding}: ✅ robust across all checks")

    return lines


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE: MASTER SUMMARY HEATMAP
# ═══════════════════════════════════════════════════════════════════════════════

def plot_robustness(results):
    findings = ["A. U-shape", "B. Pure-green\n→ failure", "C. Pure-trad\n→ exit",
                "D. Green×novelty", "E. Cross-\ncommunity", "F. Dose-response"]
    findings_keys = ["A. U-shape","B. Pure-green → failure","C. Pure-trad → exit",
                     "D. Green × novelty","E. Cross-community","F. Dose-response"]

    cols = [("Baseline","baseline"),("+ Year FE","fe"),("+ Sector FE","sector_fe"),
            ("- Food","no_food"),("-Top2 sect","no_top2"),("Americas","region_Americas"),
            ("Europe","region_Europe"),("Asia","region_Asia"),("ESG=Green","esg_green"),
            ("ESG=Trad","esg_trad"),("Leiden","leiden"),("Lvn 0.5","louvain_05"),
            ("Lvn 2.0","louvain_20"),("-MegaInv","no_mega"),("Misclass\n85%","misclass")]

    flag_to_num = {"✅":1,"⚠️":0.5,"–":0.7,"❌":0,"":0.3}
    flag_to_col = {"✅":"#27ae60","⚠️":"#f39c12","–":"#95a5a6","❌":"#e74c3c","":"#bdc3c7"}

    fig, ax = plt.subplots(figsize=(20, 5), dpi=150, facecolor="white")
    for j, (col_label, col_key) in enumerate(cols):
        for i, (finding, fkey) in enumerate(zip(findings, findings_keys)):
            b0, p0 = results["baseline"].get(fkey, (np.nan, np.nan))
            b, p   = results.get(col_key, {}).get(fkey, (np.nan, np.nan))
            fl     = flag(b0, p0, b, p) if not np.isnan(b) else "–"
            color  = flag_to_col.get(fl, "#bdc3c7")
            rect   = plt.Rectangle((j-0.45, i-0.45), 0.9, 0.9, facecolor=color,
                                    edgecolor="white", linewidth=1.5)
            ax.add_patch(rect)
            txt = fl if fl else "n/a"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7,
                    color="white" if fl in ["✅","❌"] else "black", fontweight="bold")

    ax.set_xlim(-0.5, len(cols)-0.5)
    ax.set_ylim(-0.5, len(findings)-0.5)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([c[0] for c in cols], rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(findings)))
    ax.set_yticklabels(findings, fontsize=9)
    ax.set_title("Master Robustness Table — Climate VC Paper\n"
                 "✅ Robust  ⚠️ Weakens  ❌ Drops  – Not sig. in baseline",
                 fontsize=11, fontweight="bold")

    # Legend
    from matplotlib.patches import Patch
    legend = [Patch(color="#27ae60",label="✅ Robust"),Patch(color="#f39c12",label="⚠️ Weakens"),
              Patch(color="#e74c3c",label="❌ Drops"),Patch(color="#95a5a6",label="– Not sig.")]
    ax.legend(handles=legend, loc="upper right", fontsize=8, bbox_to_anchor=(1.0,-0.15), ncol=4)

    plt.tight_layout()
    fig.savefig(OUT/"fig1_master_robustness.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  Saved fig1_master_robustness.png")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    # Initialise results container
    results = defaultdict(dict)
    for k in ["baseline","fe","sector_fe","no_food","no_top2","esg_green","esg_trad",
              "region_Americas","region_Europe","region_Asia",
              "leiden","louvain_05","louvain_10","louvain_20",
              "no_mega","misclass","cox","nonlinear","flag_fe"]:
        results[k] = {}

    all_lines = [
        "="*72,
        "FINAL PRE-SUBMISSION ROBUSTNESS AUDIT (9 BLOCKS)",
        "Company level: N=6,718  |  All models: year FE baseline",
        "="*72,
    ]

    cdf, nm, cp = load_base()

    l1, block1_res = block1(cdf, results); all_lines += l1
    l2             = block2(cdf, results); all_lines += l2
    l3             = block3(cdf, results); all_lines += l3
    l4             = block4(cdf, results); all_lines += l4
    l5             = block5(cdf, nm, results); all_lines += l5
    l6             = block6(cdf, results); all_lines += l6
    l7             = block7(cdf, nm, results); all_lines += l7
    l8, cdf        = block8(cdf, nm, results); all_lines += l8
    l9             = block9(cdf, results); all_lines += l9

    all_lines += master_table(results)
    all_lines.append(f"\n  Total runtime: {time.time()-t0:.0f}s")

    output = "\n".join(all_lines)
    print("\n" + output)
    (OUT/"results.txt").write_text(output)
    plot_robustness(results)
    print(f"\nDone → {OUT}/")


if __name__ == "__main__":
    main()
