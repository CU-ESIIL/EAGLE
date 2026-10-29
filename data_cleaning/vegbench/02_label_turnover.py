"""Label turnover at repeat-survey locations: how often does a VegBench survey get the same NVC label as an
earlier survey of the same spot, at each level of the taxonomy, as a function of the time between visits?

A revisit pair = two surveys within REVISIT_M metres (EPSG:3310) of each other and at least MIN_GAP_YEARS apart.
Uses the full benchmark (not only the lidar-matched rows), because the lidar match does not matter here.
Agreement at a level is compared only where both surveys have a label at that level.

The pairs are not independent (a site visited k times gives k(k-1)/2 pairs, and one project dominates), so
confidence intervals come from resampling whole sites (connected components of the pair graph).

Writes (catalog/ and datasets/VegBench_eval/figures/):
  turnover_pairs.csv            one row per revisit pair: ids, projects, gap, agreement flag per level
  turnover_by_gap.csv           agreement and 95% site-bootstrap interval per gap bin and level
  turnover_by_radius.csv        sensitivity of the overall agreement to the revisit radius
  turnover_by_project.csv       pairs per project and gap bin (the confound in the longer gaps)
  turnover_summary.txt          the numbers quoted in README.md
  figures/01_label_turnover.png
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
CSV = ROOT / "datasets/raw/vegbench/benchmark_splits/VegBench/VegBench.csv"
CAT = Path(__file__).parent / "catalog"
FIG = ROOT / "datasets/VegBench_eval/figures"
REVISIT_M = 5.0
SENSITIVITY_M = [1.0, 5.0, 10.0]
MIN_GAP_YEARS = 1.0
N_BOOT = 500
BINS = [1, 2, 4, 6, 11, 21, 60]
BIN_LABELS = ["1", "2-3", "4-5", "6-10", "11-20", "20+"]
LEVELS = ["survey_formation_class", "survey_formation_subclass", "survey_formation", "survey_division",
          "survey_macrogroup", "survey_group", "survey_alliance", "survey_association"]
SHORT = [l.replace("survey_", "") for l in LEVELS]


def revisit_pairs(d, tree, radius):
    p = tree.query_pairs(radius, output_type="ndarray")
    i, j = p.T
    swap = d.t.values[i] > d.t.values[j]  # i is the earlier survey
    i, j = np.where(swap, j, i), np.where(swap, i, j)
    gap = (d.t.values[j] - d.t.values[i]).astype("timedelta64[D]").astype(int) / 365.25
    o = pd.DataFrame({"i": i, "j": j, "gap_years": gap})
    o = o[o.gap_years >= MIN_GAP_YEARS].reset_index(drop=True)
    o["id_earlier"] = d.survey_point_id.values[o.i]
    o["id_later"] = d.survey_point_id.values[o.j]
    o["project_earlier"] = d.ProjectCode.values[o.i]
    o["project_later"] = d.ProjectCode.values[o.j]
    o["same_project"] = o.project_earlier == o.project_later
    for lv in LEVELS:
        a, b = d[lv].values[o.i], d[lv].values[o.j]
        o[lv] = np.where(pd.isna(a) | pd.isna(b), np.nan, (a == b).astype(float))
    o["gap_bin"] = pd.cut(o.gap_years, BINS, right=False, labels=BIN_LABELS)
    return o


def site_ids(o, n):
    m = coo_matrix((np.ones(len(o)), (o.i, o.j)), shape=(n, n))
    _, comp = connected_components(m, directed=False)
    return comp[o.i]


def by_gap(o, rng):
    """Mean agreement per gap bin and level, with a site-bootstrap 95% interval."""
    groups = {s: g.index.values for s, g in o.groupby("site")}
    sites = np.array(list(groups))
    est = o.groupby("gap_bin", observed=True)[LEVELS].mean()
    boot = []
    for _ in range(N_BOOT):
        pick = rng.choice(sites, len(sites))
        boot.append(o.loc[np.concatenate([groups[s] for s in pick])].groupby("gap_bin", observed=True)[LEVELS].mean())
    boot = pd.concat(boot)
    lo, hi = boot.groupby(level=0).quantile(0.025), boot.groupby(level=0).quantile(0.975)
    n_pairs = o.groupby("gap_bin", observed=True).size()
    n_sites = o.groupby("gap_bin", observed=True).site.nunique()
    rows = []
    for b in est.index:
        for lv in LEVELS:
            rows.append(dict(gap_bin=b, level=lv.replace("survey_", ""), n_pairs=n_pairs[b], n_sites=n_sites[b],
                             n_labeled=int(o[o.gap_bin == b][lv].notna().sum()),
                             agreement=est.loc[b, lv], ci_low=lo.loc[b, lv], ci_high=hi.loc[b, lv]))
    return pd.DataFrame(rows)


def main():
    CAT.mkdir(exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(CSV, low_memory=False)
    d["t"] = pd.to_datetime(d.SurveyDate, utc=True)
    tree = cKDTree(d[["x_3310", "y_3310"]].values)
    rng = np.random.default_rng(0)

    # sensitivity to the revisit radius
    sens = []
    for r in SENSITIVITY_M:
        p = revisit_pairs(d, tree, r)
        sens.append(dict(radius_m=r, n_pairs=len(p), n_surveys=len(set(p.i) | set(p.j)),
                         **{l.replace("survey_", ""): p[l].mean() for l in LEVELS}))
    pd.DataFrame(sens).to_csv(CAT / "turnover_by_radius.csv", index=False, float_format="%.3f")

    o = revisit_pairs(d, tree, REVISIT_M)
    o["site"] = site_ids(o, len(d))
    g = by_gap(o, rng)
    g.to_csv(CAT / "turnover_by_gap.csv", index=False, float_format="%.3f")
    o.drop(columns=["i", "j"]).to_csv(CAT / "turnover_pairs.csv", index=False, float_format="%.3f")
    top = o.project_earlier.value_counts().index[:10]
    pd.crosstab(o.project_earlier, o.gap_bin).reindex(top).to_csv(CAT / "turnover_by_project.csv")

    # alliance change vs log(gap): logistic slope with site-clustered errors
    import statsmodels.api as sm
    x = o[o.survey_alliance.notna()]
    fit = sm.GLM(1 - x.survey_alliance, sm.add_constant(np.log(x.gap_years)), family=sm.families.Binomial()).fit(
        cov_type="cluster", cov_kwds={"groups": x.site})
    ch = o[o.survey_alliance == 0]
    allp = o[LEVELS].mean().rename(lambda s: s.replace("survey_", ""))

    lines = [f"revisit radius {REVISIT_M} m, gap >= {MIN_GAP_YEARS} yr: {len(o)} pairs, {len(set(o.i) | set(o.j))} surveys, "
             f"{o.site.nunique()} sites, {o.project_earlier.nunique()} projects, same project {o.same_project.mean():.2f}",
             f"gap quantiles (yr): {o.gap_years.describe().round(1).to_dict()}",
             "overall agreement: " + ", ".join(f"{k} {v:.2f}" for k, v in allp.items()),
             "labeled pairs per level: " + ", ".join(f"{l.replace('survey_', '')} {int(o[l].notna().sum())}" for l in LEVELS),
             f"alliance change vs log(gap): slope {fit.params['gap_years']:.3f}, p {fit.pvalues['gap_years']:.3f}",
             f"of {len(ch)} pairs with a changed alliance: group also changed {(ch.survey_group == 0).mean():.2f}, "
             f"macrogroup {(ch.survey_macrogroup == 0).mean():.2f}, formation class {(ch.survey_formation_class == 0).mean():.2f}"]
    (CAT / "turnover_summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(g.pivot(index="gap_bin", columns="level", values="agreement")[SHORT].round(2).assign(
        n=o.groupby("gap_bin", observed=True).size()).to_string())

    # figure: agreement vs gap bin per level, and overall agreement per level
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [2, 1]})
    show = ["formation_class", "formation", "macrogroup", "group", "alliance", "association"]
    cm = plt.get_cmap("viridis")
    for k, lv in enumerate(show):
        s = g[(g.level == lv) & (g.gap_bin != "20+")]
        xs = np.arange(len(s)) + (k - len(show) / 2) * 0.04
        ax[0].errorbar(xs, s.agreement, yerr=[s.agreement - s.ci_low, s.ci_high - s.agreement], marker="o", ms=4,
                       lw=1.2, capsize=2, color=cm(k / (len(show) - 1)), label=lv)
    labs = [b for b in BIN_LABELS if b != "20+"]
    n = o.groupby("gap_bin", observed=True).size()
    ax[0].set_xticks(range(len(labs)), [f"{b}\n(n={n[b]})" for b in labs])
    ax[0].set_xlabel("years between surveys"), ax[0].set_ylabel("fraction with same label")
    ax[0].set_ylim(0, 1.02), ax[0].legend(fontsize=8, ncol=2, loc="lower left"), ax[0].grid(alpha=0.3)
    ax[0].set_title("Label agreement at revisited locations (95% CI over sites)")
    ax[1].barh(range(len(SHORT)), allp.values, color=[cm(k / (len(SHORT) - 1)) for k in range(len(SHORT))])
    ax[1].set_yticks(range(len(SHORT)), SHORT), ax[1].invert_yaxis(), ax[1].set_xlim(0, 1)
    ax[1].set_xlabel("overall agreement"), ax[1].set_title(f"All {len(o)} pairs")
    for k, v in enumerate(allp.values):
        ax[1].text(v + 0.01, k, f"{v:.2f}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "01_label_turnover.png", dpi=150)


if __name__ == "__main__":
    main()
