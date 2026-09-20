"""Four Runners - the interactive companion to notebooks/10_four_runners_full_story.ipynb.

Same data, same models, same six sections. The notebook stays the canonical write-up; this app
adds controls so a reader can push on the numbers instead of only reading them.

Run locally:  streamlit run streamlit_app.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from matplotlib.ticker import PercentFormatter
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------- setup ----

st.set_page_config(
    page_title="Four Runners",
    page_icon="🏃",
    layout="wide",
    initial_sidebar_state="expanded",
)

plt.style.use("seaborn-v0_8-whitegrid")
# a family LIST (not the sans-serif preference list) is what enables per-glyph fallback,
# so the VO2 subscript still renders where Arial has no glyph for it
plt.rcParams["font.family"] = ["Arial", "DejaVu Sans"]

COLORS = {0: "#5B8FF9", 1: "#61DDAA", 2: "#F6BD16", 3: "#E8684A"}
NAME_MAP = {0: "Newcomers", 1: "Regulars", 2: "Naturals", 3: "Veterans"}
ORDER = ["Newcomers", "Regulars", "Naturals", "Veterans"]
INK, GREY, GREY5 = "#273142", "#6B7280", "#8C8C8C"
RS = 0

DATA = Path(__file__).parent / "data"

BODY = [
    "age", "running_experience_months", "previous_marathon_count",
    "weekly_mileage_miles", "runs_per_week", "speed_work_sessions_per_week",
    "rest_days_per_week", "cross_training_hours_per_week",
    "resting_heart_rate_bpm", "vo2_max", "bmi", "injury_count",
]

PRED = [
    "running_experience_months", "weekly_mileage_miles", "runs_per_week",
    "speed_work_sessions_per_week", "cross_training_hours_per_week",
    "resting_heart_rate_bpm", "vo2_max", "age", "injury_count", "bmi",
    "previous_marathon_count",
]

SPOILERS = ["actual_finish_time_minutes", "target_finish_time_minutes", "goal_gap", "dnf",
            "medal", "medal_outcome", "runner_id", "personal_best_minutes"]
# one of each mirrored pair: rest_days = 7 - runs_per_week, missed workouts mirror adherence,
# VO2 max mirrors resting heart rate
DUPES = ["rest_days_per_week", "missed_workout_pct", "vo2_max"]


def show(fig, **kwargs):
    """Render a matplotlib figure and release it, so reruns don't pile up memory."""
    st.pyplot(fig, **kwargs)
    plt.close(fig)


def note(text):
    st.caption(f"**SIMULATED POPULATION** · {text}")


# ----------------------------------------------------------- data + fits ----


@st.cache_data(show_spinner="Loading the simulated population…")
def load():
    clean = pd.read_csv(DATA / "clean.csv")
    profiles = pd.read_csv(DATA / "profiles.csv")
    df = clean.merge(profiles, on="runner_id", how="inner", validate="one_to_one")
    df["profile_name"] = df["profile"].map(NAME_MAP)
    return clean, df


@st.cache_data(show_spinner=False)
def profile_table(_df):
    stats_ = (
        _df.groupby(["profile", "profile_name"], as_index=False)
          .agg(
              runners=("runner_id", "size"),
              experience_months=("running_experience_months", "mean"),
              runs_per_week=("runs_per_week", "mean"),
              weekly_miles=("weekly_mileage_miles", "mean"),
              vo2_max=("vo2_max", "mean"),
              speed_sessions=("speed_work_sessions_per_week", "mean"),
              finish_minutes=("actual_finish_time_minutes", "mean"),
              dnf_rate=("dnf", "mean"),
          )
          .sort_values("profile")
    )
    stats_["share"] = stats_["runners"] / len(_df) * 100
    programme = pd.crosstab(_df["profile"], _df["training_program"], normalize="index") * 100
    return stats_, programme


@st.cache_data(show_spinner="Projecting the training space…")
def pca_map(_df):
    X = StandardScaler().fit_transform(_df[BODY])
    xy = PCA(n_components=2).fit_transform(X)
    rng = np.random.default_rng(42)
    idx = rng.choice(len(_df), 6_000, replace=False)
    return xy[idx], _df["profile"].to_numpy()[idx]


@st.cache_data(show_spinner="Fitting the experience curve…")
def experience_fit(_d):
    e = _d.running_experience_months.values
    A = np.column_stack([np.ones(len(_d)), _d[PRED].values, e ** 2])
    beta = np.linalg.lstsq(A, _d.actual_finish_time_minutes.values, rcond=None)[0]
    return beta


def payoff(beta, months):
    """Minutes SAVED by one more YEAR of running, at a given experience level.

    beta is fitted against finish time, where faster is a smaller number, so the raw
    derivative is negative. Negate it so the figure reads as a payoff, not a penalty.
    """
    return -12 * (beta[1] + 2 * beta[-1] * months)


@st.cache_data(show_spinner="Scrambling columns to see what mattered… (one-off, ~20s)")
def generator_evidence(_clean):
    sec5 = _clean[_clean.dnf == 0].copy()
    cols = [c for c in _clean.select_dtypes("number").columns
            if c not in SPOILERS + DUPES and not c.endswith("_was_missing")]

    X, y = sec5[cols].values, sec5.actual_finish_time_minutes.values
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=RS)

    # constrained to a sum of per-variable terms: no interactions allowed
    additive = HistGradientBoostingRegressor(
        max_iter=300, random_state=RS,
        interaction_cst=[[i] for i in range(X.shape[1])]).fit(Xtr, ytr)
    unconstrained = HistGradientBoostingRegressor(max_iter=300, random_state=RS).fit(Xtr, ytr)

    imp = permutation_importance(additive, Xte, yte, n_repeats=3, random_state=RS, scoring="r2")
    importance = pd.Series(imp.importances_mean, index=cols).sort_values(ascending=False)

    pred = additive.predict(Xte)
    resid = yte - pred
    spread = (pd.Series(resid)
                .groupby(pd.qcut(pred, 5, labels=["fastest", "2", "3", "4", "slowest"]),
                         observed=True).std())

    return {
        "n_tested": len(cols),
        "n_flat": int((importance <= 0.0005).sum()),
        "importance": importance,
        "ratio": importance.iloc[0] / importance.iloc[1:].sum(),
        "r2_additive": additive.score(Xte, yte),
        "r2_mixed": unconstrained.score(Xte, yte),
        "resid": resid,
        "pred": pred,
        "spread": spread,
    }


clean, df = load()
d = df[df.dnf == 0].copy()
stats_, programme = profile_table(df)
comparison = stats_.set_index("profile")
beta = experience_fit(d)

# -------------------------------------------------------------- sidebar ----

with st.sidebar:
    st.markdown("## Four Runners")
    st.caption("A story built from 80,000 simulated runners")
    st.warning(
        "**Simulated population.** Every number here comes from a synthetic Kaggle dataset, "
        "not real racers. Descriptions of a file, never advice for a runner.",
        icon="⚠️",
    )
    st.markdown(
        """
**Sections**

1. [Meet four runners](#s1)
2. [Fifty-one minutes apart](#s2)
3. [The first year is the steep part](#s3)
4. [What else moved the clock](#s4)
5. [Then we checked whether to believe it](#s5)
6. [What a synthetic runner can tell you](#s6)
7. [References](#refs)
"""
    )
    st.divider()
    st.caption(
        "Companion to `notebooks/10_four_runners_full_story.ipynb`, which remains the full "
        "write-up. HSLU — Sport Data Analytics, FS26. Mayra, Paula, Petra and Ariella."
    )

# ----------------------------------------------------------------- title ----

st.title("Four Runners")
st.markdown("#### A story built from 80,000 simulated runners")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Simulated runners", f"{len(df):,}")
c2.metric("Descriptive profiles", df["profile"].nunique())
c3.metric("Finishers", f"{len(d):,}")
c4.metric("Fastest-to-slowest gap", f"{comparison.loc[0, 'finish_minutes'] - comparison.loc[3, 'finish_minutes']:.0f} min")

st.info(
    "Two things to keep in mind while reading: these are descriptions of a dataset, never advice "
    "for a runner, and every number is marked as simulated so it doesn't get mistaken for a fact "
    "about real marathons.",
    icon="📋",
)

# ------------------------------------------------------------ section 1 ----

st.divider()
st.header("1. Meet four runners", anchor="s1")

st.markdown(
    """
At the starting line, **80,000 simulated runners** look like they're all preparing for the same
race. Their preparation tells four different stories. These are four labels we gave to four
regions, not four kinds of runner — draw the groups a different way and the split looks
noticeably different.
"""
)

blurbs = {
    0: "Run less often than the others and mostly follow a beginner plan. The largest group.",
    1: "Run often, but each run tends to be short, so weekly distance stays low.",
    2: "Stand out for their bodies rather than their training — aerobic capacity is the outlier.",
    3: "Longest running history, most distance, most speed work, almost all on an advanced plan.",
}
extras = {
    0: lambda r: f"{programme.loc[0, 'Beginner']:.0f}% beginner programme",
    1: lambda r: f"{r.weekly_miles:.1f} miles/week",
    2: lambda r: f"VO₂ max {r.vo2_max:.0f}",
    3: lambda r: f"{programme.loc[3, 'Advanced']:.0f}% advanced programme",
}

cards = st.columns(4)
for col, row in zip(cards, stats_.itertuples()):
    with col.container(border=True):
        st.markdown(
            f"<h3 style='color:{COLORS[row.profile]};margin:0'>{row.profile_name}</h3>",
            unsafe_allow_html=True,
        )
        # metric and stats first so they line up across the four cards;
        # the variable-length blurb goes last, where ragged bottoms don't show
        st.metric("Mean finish", f"{row.finish_minutes:.0f} min")
        st.markdown(
            f"**{row.share:.1f}%** of runners  \n"
            f"{row.experience_months:.0f} months running  \n"
            f"{row.runs_per_week:.1f} runs/week  \n"
            f"{extras[row.profile](row)}"
        )
        st.caption(blurbs[row.profile])
note("All values describe the simulated population.")

st.markdown("### Memorable, but not separate")
st.markdown(
    """
The four portraits are easy to remember, but memorable isn't the same as separate. The map below
squeezes everything we know about each runner's training into two axes. If the four profiles were
truly different kinds of runner, we'd expect four separate clusters of dots. Instead the colours
blend into each other with no real gaps — a sign that these are regions on a continuum, not four
species. **Turn profiles off and on to see how arbitrary the boundaries are.**
"""
)

xy_s, prof_s = pca_map(df)
picked = st.multiselect(
    "Profiles to draw", ORDER, default=ORDER, key="pca_pick",
    help="The dots never move — only which ones are drawn. The groups sit on top of each other.",
)
opacity = st.slider("Dot opacity", 0.05, 1.0, 0.30, 0.05, key="pca_alpha")

fig, ax = plt.subplots(figsize=(12, 6))
for pid in sorted(df["profile"].unique()):
    if NAME_MAP[pid] not in picked:
        continue
    m = prof_s == pid
    ax.scatter(xy_s[m, 0], xy_s[m, 1], s=10, alpha=opacity, linewidths=0,
               color=COLORS[pid], label=NAME_MAP[pid])
ax.set_title("The profiles overlap: there are no natural gaps", fontsize=17, fontweight="bold", pad=16)
ax.set_xlabel("Combined training pattern (axis 1)")
ax.set_ylabel("Combined training pattern (axis 2)")
if picked:
    ax.legend(title="Runner profile", markerscale=2, frameon=True)
show(fig)
note("Training details only, no race results. 6,000-runner sample.")

with st.expander("Why this map is drawn from these twelve columns"):
    st.markdown(
        """
It uses the same twelve training columns the profiles were built from in `03`, so it is a fair
picture of the space they were drawn in. That list holds two near-mirror pairs —
`rest_days_per_week` is exactly `min(7 - runs_per_week, 3)`, and VO₂ max tracks resting heart rate
at r = −0.75 — which gives training frequency and aerobic capacity double weight on these axes.
`03` tested dropping them; the overlap survives either way.
"""
    )

# ------------------------------------------------------------ section 2 ----

st.divider()
st.header("2. Fifty-one minutes apart", anchor="s2")

new, vet = comparison.loc[0], comparison.loc[3]
gap = new["finish_minutes"] - vet["finish_minutes"]

st.markdown(
    f"""
The gap between the two ends of the profile continuum is large. Newcomers finish in about
**{new['finish_minutes']:.0f} minutes**, Veterans in about **{vet['finish_minutes']:.0f}** — a
difference of **{gap:.0f} minutes**.

That gap isn't just a side effect of how the groups were built. Finish time and drop-out were
never used to create the profiles — runners were grouped purely by how they trained, and only
afterwards did we look at how they did on race day. Put simply: **we sorted the simulated runners
by how they trained, and the results sorted themselves afterwards.**
"""
)

left, right = st.columns([1.45, 1])

with left:
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot([vet["finish_minutes"], new["finish_minutes"]], [0, 0],
            color="#CBD2DC", lw=8, solid_capstyle="round")
    ax.scatter(vet["finish_minutes"], 0, s=420, color=COLORS[3], zorder=3)
    ax.scatter(new["finish_minutes"], 0, s=420, color=COLORS[0], zorder=3)
    ax.text(vet["finish_minutes"], 0.12, f"Veterans\n{vet['finish_minutes']:.1f} min",
            ha="center", fontweight="bold", color=COLORS[3])
    ax.text(new["finish_minutes"], 0.12, f"Newcomers\n{new['finish_minutes']:.1f} min",
            ha="center", fontweight="bold", color=COLORS[0])
    ax.text((vet["finish_minutes"] + new["finish_minutes"]) / 2, -0.17,
            f"{gap:.1f}-minute gap", ha="center", fontsize=17, fontweight="bold", color=INK)
    ax.set_xlim(235, 315)
    ax.set_ylim(-0.35, 0.4)
    ax.set_yticks([])
    ax.set_xlabel("Mean finish time (minutes)")
    ax.set_title("The ends of the continuum finish 51 minutes apart", fontweight="bold", pad=14)
    for s in ["left", "right", "top"]:
        ax.spines[s].set_visible(False)
    show(fig)

with right:
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    rates = stats_["dnf_rate"] * 100
    bars = ax.bar(stats_["profile_name"], rates, color=[COLORS[i] for i in stats_["profile"]], width=0.62)
    ax.bar_label(bars, labels=[f"{v:.1f}%" for v in rates], padding=4, fontsize=12, fontweight="bold")
    ax.set_ylim(0, 3.9)
    ax.set_ylabel("DNF rate (%)")
    ax.tick_params(axis="x", labelsize=9)
    ax.set_title("Newcomers drop out most — but not in a clean order", fontweight="bold", pad=14)
    ax.spines[["top", "right"]].set_visible(False)
    show(fig)

note("Finish time and drop-out were not used to build the groups.")

st.markdown(
    """
**3.2% of Newcomers didn't finish**, against **1.5% of Veterans** — more than twice the rate. The
ordering isn't perfectly clean, though: Naturals actually drop out least of all (**1.4%**), a
shade below the Veterans. It's a broad tendency across the continuum, not a ranking — and it is
still just a comparison, not proof that being a "Newcomer" causes a runner to drop out.
"""
)

st.markdown("### Averages hide the full picture")
st.markdown(
    "The 51-minute gap is a distance between group averages. Below is what those averages hide. "
    "**Narrow the window or drop profiles to see how far the ranges overlap.**"
)

lo, hi = int(d.actual_finish_time_minutes.min()), int(d.actual_finish_time_minutes.max())
fc1, fc2 = st.columns([1, 1.4])
picked2 = fc1.multiselect("Profiles", ORDER, default=ORDER, key="dist_pick")
window = fc2.slider("Finish-time window (minutes)", lo, hi, (lo, hi), key="dist_window")

sub = d[d.profile_name.isin(picked2)
        & d.actual_finish_time_minutes.between(*window)]

if sub.empty:
    st.warning("No runners in that window. Widen the range or add a profile back.")
else:
    kept = [p for p in ORDER if p in picked2]
    fig, ax = plt.subplots(figsize=(13, 5.5))
    sns.violinplot(data=sub, y="profile_name", x="actual_finish_time_minutes",
                   order=kept, palette=[COLORS[ORDER.index(p)] for p in kept],
                   hue="profile_name", hue_order=kept, legend=False,
                   inner="quartile", cut=0, linewidth=1, ax=ax)
    ax.set_title("Finish-time ranges overlap across all four profiles",
                 fontsize=17, fontweight="bold", pad=16)
    ax.set_xlabel("Actual finish time (minutes)")
    ax.set_ylabel("")
    show(fig)

    m1, m2, m3 = st.columns(3)
    m1.metric("Runners shown", f"{len(sub):,}",
              f"{len(sub) / len(d) - 1:.0%} vs all finishers" if len(sub) != len(d) else None,
              delta_color="off")
    m2.metric("Median finish", f"{sub.actual_finish_time_minutes.median():.0f} min")
    m3.metric("Spread (10th–90th pct)",
              f"{sub.actual_finish_time_minutes.quantile(.9) - sub.actual_finish_time_minutes.quantile(.1):.0f} min")
    note("Finishers only. Drop-outs have no finish time and sit out of this chart.")

# ------------------------------------------------------------ section 3 ----

st.divider()
st.header("3. The first year is the steep part", anchor="s3")

st.markdown(
    """
Section 2 showed how far apart the groups finish. This section asks a more personal question: for
one runner moving through time, when does another year of running matter most? It turns out
experience doesn't pay off at a steady rate — it pays off fastest right at the start.
**Move the marker to read the payoff at any point on the curve.**
"""
)

years_in = st.slider("Experience so far (years)", 0.5, 12.0, 1.0, 0.5, key="exp_slider")
months_in = years_in * 12
value_now = payoff(beta, months_in)

k1, k2, k3 = st.columns(3)
k1.metric("One more year is worth", f"{value_now:.1f} min")
k2.metric("vs. at 6 months", f"{payoff(beta, 6):.1f} min", f"{value_now - payoff(beta, 6):+.1f} min")
k3.metric("vs. at 10 years", f"{payoff(beta, 120):.1f} min", f"{value_now - payoff(beta, 120):+.1f} min")

turning_point = -beta[1] / (2 * beta[-1]) / 12
if value_now <= 0:
    st.caption(
        f":orange[**Read this one carefully.**] Past about **{turning_point:.1f} years** the fitted "
        "payoff goes negative. That is a property of fitting a parabola — a parabola has to turn "
        "over eventually — not evidence that running longer makes a runner slower. Treat the far "
        "tail of this curve as the model running out of road, not as a finding."
    )

fig, ax = plt.subplots(figsize=(11, 4.6))
ms = np.arange(6, 145)
ax.plot(ms / 12, [payoff(beta, m) for m in ms], color=INK, lw=2.4)
ax.axhline(0, color="#CBD2DC", lw=1)
ax.scatter(years_in, value_now, s=170, color=COLORS[3], zorder=4, edgecolor="white", linewidth=1.6)
ax.annotate(f"{value_now:.1f} min", (years_in, value_now), textcoords="offset points",
            xytext=(12, 10), fontsize=13, fontweight="bold", color=COLORS[3])
ax.axvline(years_in, color=COLORS[3], lw=1, ls=":", alpha=.7)
ax.set_title("One more year is worth the most early on", fontsize=17, fontweight="bold", pad=14)
ax.set_xlabel("Years running so far")
ax.set_ylabel("Minutes gained from one more year")
show(fig)
note("The same curve applies to every runner in the file.")

st.markdown(
    """
It's tempting to read this as "training pays off less once you're experienced." That's not quite
right. Newcomers and veterans sit at different points along the *same* curve, so a newcomer's
stretch naturally looks steeper — it's the early part. The honest takeaway is about timing, not
about people: **the payoff from experience slows down for everyone, at the same rate, as time
passes.**
"""
)

with st.expander("A pattern psychology recognises — Ariella Kaeslin"):
    st.markdown(
        """
The shape of this curve — steep at the beginning, flattening out later — is not unique to running.
Newell and Rosenbloom (1981) describe the same pattern across a wide range of tasks: gains from
practice are never linear, improvement is greatest early on, and it slows continuously as skill
accumulates, regardless of the domain.

Research on habit formation tells a similar story. Lally, van Jaarsveld, Potts and Wardle (2010)
found that automaticity — how effortless and routine an action feels — grows along a curve that
levels off, and for exercise specifically a plateau is typically reached after a median of around
three months. That would place the steepest segment of the experience curve exactly in the period
when running is plausibly still turning from a deliberate decision into a stable habit.
Self-efficacy theory (Bandura, 1977) points to the same window.

One caution matters, though. The dataset is a snapshot, not a diary: it compares many different
runners at different levels of experience rather than following one runner across the years.
Dropout from running also isn't spread evenly across experience levels — a recent large cohort
study of exercise-app users found only around 10% of beginners were still active after twelve
months, against roughly 18% of intermediate and 26% of advanced exercisers (Conti et al., 2026).
If something similar holds among runners, the ten-year group in this dataset isn't simply the
six-month group grown older — it's a group that has already survived the very period when most
people who start running stop again. The flattening can be read two ways this dataset alone can't
tell apart: a genuine slowing of returns to practice, or a shift in who is even left to be
measured.
"""
    )

# ------------------------------------------------------------ section 4 ----

st.divider()
st.header("4. What else moved the clock", anchor="s4")

LABELS = {
    "speed_work_sessions_per_week": "One extra speed session per week",
    "runs_per_week": "One extra run per week",
    "weekly_mileage_miles": "One extra mile per week",
    "injury_count": "One more injury during training",
}
effects = (pd.Series(beta[1:len(PRED) + 1], index=PRED)
             .loc[list(LABELS)].rename(index=LABELS).round(1).sort_values())

st.markdown(
    "Experience matters most, but a few other everyday changes also shift the same model's "
    "prediction. Section 5 shows this file has **no interactions at all** between its inputs — "
    "every factor simply adds its own fixed number of minutes — so for this dataset the changes "
    "below really can be stacked. **Tick some and see.**"
)

pick_col, chart_col = st.columns([1, 2])

with pick_col:
    st.markdown("**Stack a few changes**")
    chosen, total = [], 0.0
    for label, val in effects.items():
        if st.checkbox(f"{label}  ({val:+.1f} min)", key=f"eff_{label}"):
            chosen.append(label)
            total += val
    if chosen:
        st.metric("Combined effect on finish time", f"{total:+.1f} min",
                  "faster" if total < 0 else ("slower" if total > 0 else "cancels out"),
                  delta_color="inverse")
    else:
        st.metric("Combined effect on finish time", "—",
                  "tick a change to stack it", delta_color="off")
    if total:
        st.caption(
            "Stacking is only legitimate because the file is purely additive — see Section 5. "
            "It is still a pattern across 80,000 simulated rows, not minutes any runner could bank."
        )

with chart_col:
    fig, ax = plt.subplots(figsize=(9, 4.4))
    bar_colors = [
        (COLORS[3] if v > 0 else COLORS[0]) if k in chosen else ("#F0C3B8" if v > 0 else "#C3D6F9")
        for k, v in effects.items()
    ]
    bars = ax.barh(effects.index, effects.values, color=bar_colors)
    ax.set_xlim(-3.5, 7.2)
    ax.bar_label(bars, labels=[f"{v:+.1f} min" for v in effects.values],
                 padding=6, fontweight="bold")
    ax.axvline(0, color=INK, lw=1)
    ax.set_xlabel("Change in predicted finish time (minutes) — negative is faster")
    ax.set_title("Four everyday changes and what they were worth",
                 fontsize=16, fontweight="bold", pad=14)
    show(fig)

note("Patterns in the data, not returns any one runner could bank.")

st.markdown(
    """
One more injury during training costs about **7 minutes**, easily the biggest single hit on the
list — bigger than a whole week of extra training buys back. Plenty of other things we measured
barely moved the prediction at all, which raises the next question: how much of any of this should
we actually believe?
"""
)

# ------------------------------------------------------------ section 5 ----

st.divider()
st.header("5. Then we checked whether to believe it", anchor="s5")
st.markdown(
    "Everything so far took the dataset at its word. This section does the opposite — it asks the "
    "file to explain itself."
)

ev = generator_evidence(clean)

st.markdown("### Most of what we measured turned out not to matter")
st.markdown(
    """
The file records dozens of things — sleep, nutrition, motivation, recovery, how often someone
shows up to club runs. To find out what each was worth, we built a model that predicts finish time
from everything at once, then scrambled one column at a time to see how much worse the model got.
If scrambling a column barely hurts it, that column wasn't doing much in the first place.
"""
)

e1, e2, e3 = st.columns(3)
e1.metric("Things tested", ev["n_tested"])
e2.metric("Made almost no difference", ev["n_flat"], f"{ev['n_flat'] / ev['n_tested']:.0%} of them")
e3.metric("Experience vs. everything else", f"{ev['ratio']:.0f}×")

top_n = st.slider("How many columns to show", 5, min(20, ev["n_tested"]), 8, key="imp_n")
top = ev["importance"].head(top_n)[::-1]

fig, ax = plt.subplots(figsize=(11, 0.34 * top_n + 1.4))
ax.barh(range(len(top)), top.values,
        color=[COLORS[0] if v > 0.01 else GREY5 for v in top.values])
for i, v in enumerate(top.values):
    ax.text(v + .02, i, f"{v:.4f}", va="center", fontsize=9, color="#3C3C3C")
ax.set_yticks(range(len(top)))
ax.set_yticklabels([c.replace("_", " ") for c in top.index], fontsize=10)
ax.set_xlim(0, max(top.values) * 1.28)
ax.set_xlabel("how much accuracy drops when the column is scrambled")
ax.set_title("One thing does almost all the work", loc="left", fontweight="bold")
show(fig)

st.markdown(
    f"""
Running experience is worth roughly **{ev['ratio']:.0f} times** as much as everything else put
together. Injuries and resting heart rate add a little on top. Below that,
**{ev['n_flat']} of the {ev['n_tested']} things we tested made almost no difference at all**.

Weekly mileage is one of those columns, and that detail gives the file away. In real marathon
research, weekly mileage is one of the single strongest predictors of finish time there is. A
dataset where it does almost nothing isn't just an unusual dataset about running — it looks like
it isn't really about running at all.
"""
)

st.markdown("### What actually built the file")

t1, t2, t3 = st.tabs([
    "① No teamwork between factors",
    "② The leftover is pure static",
    "③ The numbers look manufactured",
])

with t1:
    st.markdown(
        "We tested whether letting the model combine factors in clever ways — say, mileage "
        "mattering more for veterans than for newcomers — made any real difference. It didn't."
    )
    a, b, c = st.columns(3)
    a.metric("Every factor kept separate", f"{ev['r2_additive']:.1%}", "of the variation explained")
    b.metric("Allowed to mix factors", f"{ev['r2_mixed']:.1%}", "of the variation explained")
    c.metric("Gain from mixing", f"{ev['r2_mixed'] - ev['r2_additive']:+.2%}", "essentially nothing")
    st.markdown(
        "Every input in this file simply adds or subtracts its own fixed number of minutes, "
        "independently of everything else. That is what makes the stacking in Section 4 legitimate "
        "here — and it is not how real training works."
    )

with t2:
    resid = ev["resid"]
    st.markdown(
        f"After accounting for experience, injuries and the rest, there's still a gap between the "
        f"model's guess and the real number — averaging about **{resid.std():.0f} minutes** in "
        f"either direction. That gap has no pattern to it."
    )
    a, b, c = st.columns(3)
    a.metric("Typical leftover gap", f"{resid.std():.0f} min")
    b.metric("Lopsided?", f"{stats.skew(resid):+.2f}", "0 = perfectly balanced", delta_color="off")
    c.metric("Heavy tails?", f"{stats.kurtosis(resid):+.2f}", "0 = plain bell curve", delta_color="off")

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.hist(resid, bins=60, density=True, color=COLORS[0], alpha=.55,
            edgecolor="white", linewidth=.4)
    grid = np.linspace(resid.min(), resid.max(), 300)
    ax.plot(grid, stats.norm.pdf(grid, resid.mean(), resid.std()), color=COLORS[3], lw=2,
            label=f"random noise, {resid.std():.0f} min typical size")
    ax.set_xlabel("minutes the model got wrong by")
    ax.set_yticks([])
    ax.legend(frameon=False)
    ax.set_title("The leftover looks exactly like noise", loc="left", fontweight="bold")
    show(fig)

    st.markdown("**And it's just as large for fast runners as for slow ones:**")
    st.dataframe(
        ev["spread"].round(2).rename("typical gap size (minutes)").to_frame().T,
        width="stretch",
    )

with t3:
    st.markdown(
        "Real training data is messy — no two runners report exactly the same weekly mileage down "
        "to the decimal. In this file, that is not what happens."
    )
    BOUNDED = ["vo2_max", "resting_heart_rate_bpm", "bmi", "sleep_hours_avg",
               "training_adherence_pct", "weekly_mileage_miles", "long_run_distance_km"]
    bounds = pd.DataFrame({
        "min": clean[BOUNDED].min(),
        "max": clean[BOUNDED].max(),
        "share stuck at the minimum": (clean[BOUNDED] == clean[BOUNDED].min()).mean(),
    }).sort_values("share stuck at the minimum")

    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.barh(range(len(bounds)), bounds["share stuck at the minimum"].values,
            color=[COLORS[3] if v > .2 else GREY5 for v in bounds["share stuck at the minimum"]])
    ax.set_yticks(range(len(bounds)))
    ax.set_yticklabels([c.replace("_", " ") for c in bounds.index], fontsize=10)
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlabel("share of runners sharing the exact same lowest value")
    ax.set_title("Numbers that look manufactured", loc="left", fontweight="bold")
    show(fig)

    st.dataframe(bounds.style.format({"min": "{:.1f}", "max": "{:.1f}",
                                      "share stuck at the minimum": "{:.0%}"}),
                 width="stretch")

    st.markdown(
        """
Half of all runners share the *exact* same weekly mileage, and four out of five share the exact
same longest training run. That's what you'd expect from generating numbers by computer and
cutting off any that went past a limit.

That also closes the loop on the first clue. Weekly mileage didn't look like it mattered partly
because, for half the file, it doesn't vary at all — a column pinned to its floor for 40,000
runners leaves a model very little variation to find an effect in. The flat importance score and
the manufactured-looking numbers aren't two separate oddities; the second helps produce the first.
"""
    )

st.markdown("### Put together")
st.success(
    "**One sentence covers it:** build each runner's stats independently and cut off the extremes, "
    "work out a finish time that's mostly just experience plus a few small adjustments, sprinkle "
    "about 21 minutes of pure randomness on top, then quietly set a target time by shaving a bit "
    "off the real result.",
    icon="🧪",
)
st.markdown(
    f"""
That one sentence explains everything that seemed odd. Most of what we measured didn't matter
because it was never part of the formula. The four profiles don't behave differently from each
other because there's no teamwork between factors for a formula like this to produce. No model
gets much closer than it already does — about 18 minutes of *average* error in `05`, against the
{ev['resid'].std():.0f}-minute spread above — because roughly {ev['resid'].std():.0f} minutes of
every runner's time comes straight from a random number generator. And nobody beats their own
target by much, because the target was built from the result instead of set before the race.

None of this makes Sections 1 to 4 wrong — they describe the file honestly. It just means the file
is describing itself, not marathon running.

*(For what it's worth: real studies of marathon training consistently find weekly mileage to be
one of the strongest predictors of finish time there is — the opposite of what this file shows
(Doherty et al., 2020).)*
"""
)

# ------------------------------------------------------------ section 6 ----

st.divider()
st.header("6. What a synthetic runner can and cannot tell you", anchor="s6")

st.markdown(
    """
Step back, and the dataset produced something that feels believable: four distinct-looking types
of runner, a real gap between the fastest and slowest, a drop-out pattern that lines up with less
experience, and an improvement curve that's steep at first and flattens out later. Every one of
those shapes looks like what you'd expect from real marathon data.

Section 5 is why none of it can be trusted as a fact about running. More than half of what we
measured had no real bearing on finish time, and the file gets one of running's best-established
truths backwards. A file that gets that wrong isn't a file that happens to disagree with the
science — it's a file that was never built to agree with it.

That's not a flaw to apologise for — it's what this kind of dataset is for. A well-built synthetic
population can teach you what a marathon story looks like: the shape of a group of runners, the
shape of a drop-out curve, the shape of getting better over time. It cannot teach you what's
actually true about marathon running, because the truth about running was never built into how the
numbers were generated.
"""
)

st.info(
    "**A synthetic dataset can teach you what a story looks like. It cannot tell you what is "
    "true.** Both halves of that matter, and neither one cancels the other out.",
    icon="🎯",
)

# --------------------------------------------------------------- refs ----

st.divider()
st.subheader("References", anchor="refs")
st.markdown(
    """
Bandura, A. (1977). Self-efficacy: Toward a unifying theory of behavioral change. *Psychological
Review*, 84(2), 191–215. https://doi.org/10.1037/0033-295X.84.2.191

Conti, F., Marzagão, T., Galpin, A. J., & Schoenfeld, B. J. (2026). Predictors of long-term
resistance exercise adherence: Evidence from a large cohort of mobile app users of various
experience levels. *Frontiers in Sports and Active Living*, 8.
https://doi.org/10.3389/fspor.2026.1855668

Doherty, C., Keogh, A., Davenport, J., Lawlor, A., Smyth, B., & Caulfield, B. (2020). An
evaluation of the training determinants of marathon performance: a meta-analysis with
meta-regression. *Journal of Science and Medicine in Sport*, 23(2), 182–188.
https://doi.org/10.1016/j.jsams.2019.09.013

Lally, P., van Jaarsveld, C. H. M., Potts, H. W. W., & Wardle, J. (2010). How are habits formed:
Modelling habit formation in the real world. *European Journal of Social Psychology*, 40(6),
998–1009. https://doi.org/10.1002/ejsp.674

Newell, A., & Rosenbloom, P. S. (1981). Mechanisms of skill acquisition and the law of practice.
In J. R. Anderson (Ed.), *Cognitive skills and their acquisition* (pp. 1–55). Lawrence Erlbaum
Associates.
"""
)

st.caption(
    "HSLU — Sport Data Analytics, FS26 · Mayra, Paula, Petra and Ariella · "
    "Full write-up: `notebooks/10_four_runners_full_story.ipynb`"
)
