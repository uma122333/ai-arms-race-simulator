"""
AI Arms Race & Governance Simulator
Recreating Armstrong, Bostrom & Shulman (2016), "Racing to the precipice:
a model of artificial intelligence development", plus an Expected-Value
governance overlay (the "Weakest Link" chain  dP = a * c * w * r).

Run with:   streamlit run app.py
"""

import numpy as np
import plotly.graph_objects as go
import streamlit as st

# ---- MATH START ------------------------------------------------------------
# Model: n teams, capabilities drawn uniformly from [0, mu]; enmity e in (0, 1].
# Every function returns the overall AI-disaster probability (1 - s of the
# winning team), clamped to [0, 1].

EPS = 1e-12  # guards against division by zero


def clamp01(x: float) -> float:
    """Clamp a probability to [0, 1]."""
    return float(min(1.0, max(0.0, x)))


def risk_no_info(mu: float, e: float, n: int) -> float:
    """No information: s = mu/(e n) if mu < e n, else 1. Risk = 1 - s."""
    denom = max(e * n, EPS)
    if mu >= denom:
        return 0.0
    return clamp01(1.0 - mu / denom)


def risk_private(mu: float, e: float, n: int) -> float:
    """
    Private information: each team knows only its own capability x and plays
    s(x) = x / K with K = e n - e + 1 (capped at 1). Integrating over the
    winner's capability (density n x^(n-1) / mu^n) gives:
        mu <  K:  1 - n*mu / ((n+1) K)
        mu >= K:  K^n / ((n+1) mu^n)
    """
    K = max(e * n - e + 1.0, EPS)
    mu = max(mu, EPS)
    if mu < K:
        return clamp01(1.0 - n * mu / ((n + 1) * K))
    return clamp01(K**n / ((n + 1) * mu**n))


def risk_public(mu: float, e: float, n: int) -> float:
    """
    Public information: leader's safety s = Delta/e (capped at 1), where
    Delta is the capability gap between the top two teams.
        mu <  e:  1 - mu / (e (n+1))
        mu >= e:  (mu-e)^(n+1) / (e (n+1) mu^n) - mu / (e (n+1)) + 1
    """
    e = max(e, EPS)
    mu = max(mu, EPS)
    if mu < e:
        return clamp01(1.0 - mu / (e * (n + 1)))
    return clamp01((mu - e) ** (n + 1) / (e * (n + 1) * mu**n)
                   - mu / (e * (n + 1)) + 1.0)


def lives_saved(risk: float, dP: float, N: float, mode: str) -> float:
    """
    Expected lives saved by governance.
    mode "spec":     Risk * dP * N   (formula as specified in the project brief)
    mode "absolute": min(dP, Risk) * N  (treats r as an absolute risk cut)
    """
    if mode == "absolute":
        return min(clamp01(dP), clamp01(risk)) * N
    return clamp01(risk) * clamp01(dP) * N
# ---- MATH END --------------------------------------------------------------


def fmt_lives(x: float) -> str:
    """Human-readable large numbers."""
    for scale, name in ((1e12, "trillion"), (1e9, "billion"), (1e6, "million")):
        if abs(x) >= scale:
            return f"{x / scale:,.2f} {name}"
    return f"{x:,.0f}"


# ---------------------------------------------------------------------------
# Page + sidebar
# ---------------------------------------------------------------------------
st.set_page_config(page_title="AI Arms Race & Governance Simulator", layout="wide")
st.title("AI Arms Race & Governance Simulator")
st.caption("Recreating Armstrong, Bostrom, & Shulman (2016) with an "
           "Expected-Value governance overlay")

with st.sidebar:
    st.header("Model Parameters (Armstrong et al.)")
    n = st.slider("Number of competitors (n)", 2, 10, 2, step=1)
    e = st.slider("Enmity / geopolitical tension (e)", 0.01, 1.0, 1.0, step=0.01)
    mu = st.slider("Capability range / importance (μ)", 0.1, 10.0, 2.0, step=0.1)

    st.header("EV & Risk Parameters")
    pop_choice = st.radio("Population at stake (N)",
                          ["Present Earth population (8×10⁹)",
                           "Future generations (10¹⁴)"])
    N = 8e9 if pop_choice.startswith("Present") else 1e14
    baseline = st.number_input("Baseline disaster risk probability",
                               0.0, 1.0, 0.05, step=0.01,
                               help="Default 5%: median expert estimate, "
                                    "Grace et al. (2024).")
    regime = st.selectbox("Regime used for the EV cards",
                          ["No information", "Private information",
                           "Public information"], index=1)
    risk_source = st.radio("Risk used in EV calculation",
                           ["Armstrong model (selected regime)",
                            "Baseline risk"])
    saved_mode_label = st.radio(
        "Lives-saved formula",
        ["As specified: Risk × ΔP × N",
         "Absolute: min(ΔP, Risk) × N"],
        help="If r is an *absolute* risk reduction, multiplying by Risk again "
             "double-counts. Switch to compare.")
    saved_mode = "spec" if saved_mode_label.startswith("As specified") else "absolute"

    st.header("Governance Chain  ΔP = a·c·w·r")
    a = st.slider("a — agreement chance", 0.0, 1.0, 0.3, step=0.01)
    c = st.slider("c — compliance chance (enforcement q)", 0.0, 1.0, 0.5, step=0.01)
    w = st.slider("w — technical efficacy", 0.0, 1.0, 0.5, step=0.01)
    r = st.slider("r — absolute risk reduction", 0.001, 0.05, 0.01, step=0.001,
                  format="%.3f")

dP = a * c * w * r

# ---------------------------------------------------------------------------
# Computations at the current parameters
# ---------------------------------------------------------------------------
risks = {
    "No information": risk_no_info(mu, e, n),
    "Private information": risk_private(mu, e, n),
    "Public information": risk_public(mu, e, n),
}
model_risk = risks[regime]
ev_risk = model_risk if risk_source.startswith("Armstrong") else clamp01(baseline)
at_risk = ev_risk * N
saved = lives_saved(ev_risk, dP, N, saved_mode)

# ---------------------------------------------------------------------------
# Metric cards
# ---------------------------------------------------------------------------
with st.container(border=True):
    st.subheader("1 · Disaster risk by information regime")
    cols = st.columns(3)
    for col, (name, val) in zip(cols, risks.items()):
        col.metric(name, f"{val * 100:.1f}%")

c2, c3 = st.columns(2)
with c2.container(border=True):
    st.subheader("2 · Expected human lives at risk")
    st.metric(f"Risk × N  ({regime if risk_source.startswith('Armstrong') else 'baseline'})",
              fmt_lives(at_risk), help=f"Risk used: {ev_risk * 100:.2f}%  |  N = {N:.0e}")
with c3.container(border=True):
    st.subheader("3 · Expected lives saved by enforcement")
    st.metric(f"ΔP = {dP:.5f}", fmt_lives(saved),
              help="ΔP = a × c × w × r")

# ---------------------------------------------------------------------------
# Primary chart: risk vs mu (cf. Figure 1 of the paper)
# ---------------------------------------------------------------------------
mus = np.linspace(0.1, 10.0, 300)
curves = [
    ("No capability information", risk_no_info, "solid"),
    ("Private capability information", risk_private, "dash"),
    ("Public capability information", risk_public, "dot"),
]
fig = go.Figure()
colors = ["#1f77b4", "#d62728", "#2ca02c"]
for (label, fn, dash), color in zip(curves, colors):
    fig.add_trace(go.Scatter(x=mus, y=[fn(m, e, n) for m in mus], mode="lines",
                             name=label, line=dict(dash=dash, width=3, color=color)))
    fig.add_trace(go.Scatter(x=[mu], y=[fn(mu, e, n)], mode="markers",
                             marker=dict(size=11, color=color, line=dict(width=1, color="black")),
                             showlegend=False,
                             hovertemplate=f"{label}<br>μ=%{{x:.2f}}<br>risk=%{{y:.3f}}<extra></extra>"))
fig.update_layout(
    title=f"AI risk level vs. importance of capability (n={n}, e={e:.2f})",
    xaxis_title="Relative importance of capability (μ)",
    yaxis_title="AI risk level (1 − s)",
    yaxis=dict(range=[0, 1.02]), xaxis=dict(range=[0.1, 10]),
    legend=dict(orientation="h", y=-0.2), height=470,
)
st.plotly_chart(fig)

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["Information hazard comparison",
                            "Tipping point matrix (q-threshold)",
                            "Paper equations & references"])

with tab1:
    names = list(risks.keys())
    vals = [risks[k] for k in names]
    bar = go.Figure(go.Bar(
        x=names, y=vals, marker_color=colors,
        text=[f"{v * 100:.1f}%" for v in vals], textposition="outside"))
    bar.update_layout(title=f"Disaster risk by regime at μ={mu:.1f}, e={e:.2f}, n={n}",
                      yaxis=dict(title="Disaster risk (1 − s)", range=[0, 1.1]),
                      height=420)
    st.plotly_chart(bar)
    d1 = (vals[1] - vals[0]) * 100
    d2 = (vals[2] - vals[0]) * 100
    st.write(f"Relative to no information: private information shifts risk by "
             f"**{d1:+.1f} pp**, public information by **{d2:+.1f} pp**.")
    if vals[0] <= vals[1] <= vals[2] + 1e-9:
        st.info("At these settings, more information means higher disaster risk.")
    else:
        st.info("At these settings the ordering is not monotonic — try other "
                "values of μ, e, n. The information hazard is not universal.")

with tab2:
    st.markdown("Does enforcement push teams from **Race** (defect) to "
                "**Restrain** (cooperate)? Cheating pays only if "
                "(1−q)·Gain > q·Penalty, so the threshold is "
                "q* = Gain / (Gain + Penalty).")
    ratio = st.number_input("Penalty / Gain ratio", 0.1, 10.0, 1.0, step=0.1,
                            help="1.0 gives the q* = 0.5 threshold used in the essay.")
    q_star = 1.0 / (1.0 + ratio)
    q = c
    if q > q_star:
        st.success(f"q = c = {q:.2f} > q* = {q_star:.2f}  →  **Restrain**: "
                   "enforcement outweighs the gain from cheating.")
    else:
        st.error(f"q = c = {q:.2f} ≤ q* = {q_star:.2f}  →  **Race**: "
                 "cheating is still worth it.")

    cs = np.linspace(0, 1, 41)
    as_ = np.linspace(0, 1, 41)
    z = np.array([[lives_saved(ev_risk, aa * cc * w * r, N, saved_mode) for cc in cs]
                  for aa in as_])
    heat = go.Figure(go.Heatmap(x=cs, y=as_, z=z, colorscale="Viridis",
                                colorbar=dict(title="Expected lives saved"),
                                hovertemplate="c=%{x:.2f}<br>a=%{y:.2f}<br>saved=%{z:.3e}<extra></extra>"))
    heat.add_vline(x=q_star, line_dash="dash", line_color="white",
                   annotation_text=f"q* = {q_star:.2f}", annotation_font_color="white")
    heat.add_trace(go.Scatter(x=[c], y=[a], mode="markers", name="You are here",
                              marker=dict(size=13, color="red", line=dict(width=2, color="white"))))
    heat.add_annotation(x=q_star / 2, y=0.95, text="RACE", showarrow=False,
                        font=dict(color="white", size=16))
    heat.add_annotation(x=(1 + q_star) / 2, y=0.95, text="RESTRAIN", showarrow=False,
                        font=dict(color="white", size=16))
    heat.update_layout(title="Expected lives saved across enforcement (c = q) and agreement (a)",
                       xaxis_title="c — compliance / enforcement q",
                       yaxis_title="a — agreement chance", height=480)
    st.plotly_chart(heat)

with tab3:
    st.markdown("### Armstrong, Bostrom & Shulman (2016)")
    st.markdown("Teams have capability drawn uniformly from [0, μ], choose safety "
                "s ∈ [0, 1], and the disaster probability is 1 − s of the winner. "
                "e is enmity between teams.")
    st.markdown("**No information**")
    st.latex(r"s=\min\!\left(1,\tfrac{\mu}{e n}\right),\quad \text{Risk}=\max\!\left(0,\,1-\tfrac{\mu}{en}\right)")
    st.markdown("**Private information**, with K = en − e + 1")
    st.latex(r"s(x)=\min\!\left(1,\tfrac{x}{K}\right)")
    st.latex(r"\text{Risk}=\begin{cases}1-\dfrac{n\mu}{(n+1)K} & \mu<K\\[2mm]\dfrac{K^{n}}{(n+1)\mu^{n}} & \mu\ge K\end{cases}")
    st.markdown("**Public information**")
    st.latex(r"s_{top}(\Delta)=\min\!\left(1,\tfrac{\Delta}{e}\right)")
    st.latex(r"\text{Risk}=\begin{cases}1-\dfrac{\mu}{e(n+1)} & \mu<e\\[2mm]\dfrac{(\mu-e)^{n+1}}{e(n+1)\mu^{n}}-\dfrac{\mu}{e(n+1)}+1 & \mu\ge e\end{cases}")
    st.markdown("### EV governance chain (\"The Weakest Link\")")
    st.latex(r"\Delta P = a\cdot c\cdot w\cdot r")
    st.latex(r"\text{Lives at risk}=\text{Risk}\times N,\qquad \text{Lives saved}=\text{Risk}\times\Delta P\times N")
    st.markdown("### References")
    st.markdown("- Armstrong, S., Bostrom, N., & Shulman, C. (2016). Racing to the precipice: "
                "a model of artificial intelligence development. *AI & Society, 31*(2), 201–206.\n"
                "- Grace, K., et al. (2024). Thousands of AI authors on the future of AI.\n"
                "- *The Weakest Link: Why AI Is So Hard to Regulate Globally* (your essay).")
