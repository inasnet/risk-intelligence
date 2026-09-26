
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from scipy.stats import spearmanr

from mcdm_methods import (
    ahp, bwm, entropy, critic, ranking_table
)


# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Risk Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded"
)

RISKS = [
    "Credit risk",
    "Market risk",
    "Liquidity risk",
    "Operational risk",
    "Cybersecurity risk"
]

CRITERIA = [
    "Probability",
    "Financial impact",
    "Detection difficulty",
    "Recovery time"
]

RISK_INFO = {
    "Credit risk": (
        "landmark",
        "Potential losses caused by counterparty default."
    ),
    "Market risk": (
        "trending-up",
        "Exposure to financial market fluctuations."
    ),
    "Liquidity risk": (
        "wallet",
        "Difficulty meeting payment obligations."
    ),
    "Operational risk": (
        "settings",
        "Failures of processes, people or systems."
    ),
    "Cybersecurity risk": (
        "shield",
        "Cyberattacks and information-security incidents."
    )
}

PAGES = [
    "Overview",
    "Risk Assessment",
    "Criteria Weighting",
    "Risk Ranking",
    "Comparison & Sensitivity"
]


# ============================================================
# DESIGN — ONE CLEAN STYLESHEET
# ============================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');

:root {
    --navy: #101B2C;
    --navy-light: #1A2C42;
    --emerald: #36B89A;
    --emerald-light: #EAF8F3;
    --background: #F5F7FA;
    --text: #192B43;
    --muted: #748398;
    --border: #E5EAF0;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: var(--background);
    color: var(--text);
}

.block-container {
    max-width: 1500px;
    padding: 2rem 2.5rem 4rem;
}

.main h1, .main h2, .main h3 {
    color: var(--text) !important;
    letter-spacing: -0.5px;
}

.main p, .main label {
    color: #52647B;
}

/* SIDEBAR */

section[data-testid="stSidebar"] {
    background: var(--navy) !important;
    border-right: 1px solid #25374C;
    width: 255px !important;
    min-width: 255px !important;
}

section[data-testid="stSidebar"] > div {
    background: var(--navy) !important;
}

section[data-testid="stSidebar"]
[data-testid="stSidebarUserContent"] {
    padding: 26px 13px !important;
}

.sidebar-brand {
    display: flex;
    align-items: center;
    gap: 11px;
    padding: 9px 8px 34px;
}

.sidebar-logo {
    width: 42px;
    height: 42px;
    flex-shrink: 0;
    background: #20433F;
    border: 1px solid #32695E;
    border-radius: 12px;
    color: #82E6C5;
    font-size: 24px;
    display: flex;
    align-items: center;
    justify-content: center;
}

.sidebar-name {
    color: #FFFFFF;
    font-size: 14px;
    font-weight: 800;
    line-height: 1.25;
    letter-spacing: 0.3px;
}

.sidebar-subtitle {
    color: #8FA4B9;
    font-size: 9px;
    letter-spacing: 0.6px;
    margin-top: 5px;
}

.sidebar-label {
    color: #788FA7;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1.5px;
    margin: 0 0 12px 14px;
}

/* SIDEBAR BUTTONS */

section[data-testid="stSidebar"] div.stButton {
    margin-bottom: 3px;
}

section[data-testid="stSidebar"] div.stButton > button {
    background: transparent !important;
    color: #B9C8D8 !important;
    border: 0 !important;
    border-radius: 9px !important;
    box-shadow: none !important;
    width: 100%;
    min-height: 43px;
    justify-content: flex-start !important;
    text-align: left !important;
    padding: 0 12px !important;
    font-size: 12px;
    font-weight: 500;
}

section[data-testid="stSidebar"] div.stButton > button p {
    color: inherit !important;
    font-size: 12px !important;
}

section[data-testid="stSidebar"] div.stButton > button:hover {
    background: #1A2E43 !important;
    color: white !important;
}

section[data-testid="stSidebar"]
div.stButton > button[kind="primary"] {
    background: #1B4140 !important;
    color: #FFFFFF !important;
    border-left: 3px solid #72D9B0 !important;
    font-weight: 700;
}

/* SIDEBAR PROGRESS */

.sidebar-progress {
    background: #1A2C42;
    border: 1px solid #2C4056;
    border-radius: 13px;
    padding: 17px;
    margin-top: 27px;
}

.sidebar-progress-title {
    color: #C8D7E5;
    font-size: 12px;
    font-weight: 600;
}

.sidebar-progress-value {
    color: #FFFFFF;
    font-size: 28px;
    font-weight: 800;
    margin: 10px 0;
}

.sidebar-progress-track {
    background: #34495E;
    height: 5px;
    border-radius: 20px;
    overflow: hidden;
}

.sidebar-progress-fill {
    height: 5px;
    background: #72D9B0;
    border-radius: 20px;
}

.sidebar-progress-caption {
    color: #8FA4B9;
    font-size: 10px;
    margin-top: 11px;
}

/* MAIN HEADER */

.page-eyebrow {
    color: #8390A2;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1.5px;
    margin-bottom: 7px;
}

.page-title {
    color: #192B43;
    font-size: 34px;
    font-weight: 800;
    letter-spacing: -1px;
    margin-bottom: 5px;
}

.page-subtitle {
    color: #748398;
    font-size: 14px;
    margin-bottom: 27px;
}

/* METRICS */

div[data-testid="stMetric"] {
    background: #FFFFFF !important;
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 19px 21px;
    box-shadow: 0 3px 14px rgba(20,40,65,0.025);
}

div[data-testid="stMetricLabel"],
div[data-testid="stMetricLabel"] *,
div[data-testid="stMetric"] label {
    color: #748398 !important;
    opacity: 1 !important;
    font-size: 12px !important;
    font-weight: 600 !important;
}

div[data-testid="stMetricValue"],
div[data-testid="stMetricValue"] * {
    color: #192B43 !important;
    opacity: 1 !important;
    font-size: 30px !important;
    font-weight: 800 !important;
}

/* SECTION TITLES */

.section-title {
    color: #192B43;
    font-size: 19px;
    font-weight: 800;
    margin: 28px 0 15px;
}

/* RISK PORTFOLIO */

.portfolio {
    background: #FFFFFF;
    border: 1px solid var(--border);
    border-radius: 15px;
    padding: 9px 22px;
}

.portfolio-row {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 15px 0;
    border-bottom: 1px solid #EEF1F5;
}

.portfolio-row:last-child {
    border-bottom: none;
}

.portfolio-icon {
    width: 35px;
    height: 35px;
    flex-shrink: 0;
    border-radius: 10px;
    background: #EAF8F3;
    color: #15856E;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 18px;
}

.portfolio-text {
    flex: 1;
}

.portfolio-name {
    color: #26374D;
    font-size: 13px;
    font-weight: 700;
}

.portfolio-description {
    color: #8290A2;
    font-size: 11px;
    margin-top: 3px;
}

.portfolio-status {
    background: #F0F3F7;
    color: #7A8799;
    font-size: 10px;
    border-radius: 20px;
    padding: 6px 11px;
}

.portfolio-status.complete {
    background: #E7F7F0;
    color: #16856C;
}

/* MAIN BUTTONS */

.main .stButton > button[kind="primary"] {
    background: #168B77 !important;
    border: none !important;
    color: #FFFFFF !important;
    border-radius: 9px !important;
    font-weight: 700;
}

.main .stButton > button[kind="primary"]:hover {
    background: #0C6F5E !important;
}

/* TABS AND TABLES */

div[data-testid="stDataFrame"] {
    border: 1px solid var(--border);
    border-radius: 12px;
    overflow: hidden;
}

.stTabs [data-baseweb="tab-highlight"] {
    background: #168B77;
}

.stTabs [aria-selected="true"] {
    color: #168B77 !important;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# SESSION STATE
# ============================================================

if "matrix" not in st.session_state:
    st.session_state.matrix = pd.DataFrame(
        np.nan,
        index=RISKS,
        columns=CRITERIA
    )

if "weights" not in st.session_state:
    st.session_state.weights = {}

if "page" not in st.session_state:
    st.session_state.page = "Overview"


def current_matrix():
    return st.session_state.matrix.reindex(
        index=RISKS,
        columns=CRITERIA
    )


def get_matrix():
    df = current_matrix()

    if df.isna().any().any():
        raise ValueError("Complete all 20 evaluations first.")

    x = df.to_numpy(dtype=float)

    if not np.isfinite(x).all():
        raise ValueError("All scores must be finite numbers.")

    if (x < 1).any() or (x > 10).any():
        raise ValueError("Scores must be between 1 and 10.")

    return x


def get_weights(method):
    saved = st.session_state.weights[method]

    if saved["matrix"] is not None:
        if not saved["matrix"].equals(current_matrix()):
            raise ValueError(
                "Assessment data changed. "
                f"Recalculate {method} weights."
            )

    return saved["values"]


def save_weights(method, values, data_dependent=False):
    st.session_state.weights[method] = {
        "values": np.asarray(values, dtype=float),
        "matrix": (
            current_matrix().copy()
            if data_dependent else None
        )
    }


def header(title, subtitle):
    st.markdown(
        '<div class="page-eyebrow">'
        'RISK INTELLIGENCE / WORKSPACE'
        '</div>'
        f'<div class="page-title">{title}</div>'
        f'<div class="page-subtitle">{subtitle}</div>',
        unsafe_allow_html=True
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        '<div class="sidebar-brand">'
        '<div class="sidebar-logo">◈</div>'
        '<div>'
        '<div class="sidebar-name">RISK<br>INTELLIGENCE</div>'
        '<div class="sidebar-subtitle">'
        'FINANCIAL DECISION SUPPORT'
        '</div>'
        '</div>'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sidebar-label">WORKSPACE</div>',
        unsafe_allow_html=True
    )

    NAVIGATION = [
        ("▦", "Overview"),
        ("◫", "Risk Assessment"),
        ("☷", "Criteria Weighting"),
        ("▥", "Risk Ranking"),
        ("◈", "Comparison & Sensitivity")
    ]

    for symbol, destination in NAVIGATION:
        active = st.session_state.page == destination

        if st.button(
            f"{symbol}   {destination}",
            key=f"navigation_{destination}",
            type="primary" if active else "secondary",
            use_container_width=True
        ):
            st.session_state.page = destination
            st.rerun()

    completed = int(current_matrix().notna().sum().sum())
    percentage = int(completed / 20 * 100)

    progress_html = (
        '<div class="sidebar-progress">'
        '<div class="sidebar-progress-title">'
        'Assessment progress</div>'
        f'<div class="sidebar-progress-value">{percentage}%</div>'
        '<div class="sidebar-progress-track">'
        f'<div class="sidebar-progress-fill" '
        f'style="width:{percentage}%"></div>'
        '</div>'
        '<div class="sidebar-progress-caption">'
        f'{completed} of 20 scores entered'
        '</div></div>'
    )

    st.markdown(
        progress_html,
        unsafe_allow_html=True
    )

    st.caption("MCDM Research Project")


page = st.session_state.page


# ============================================================
# OVERVIEW
# ============================================================

if page == "Overview":

    header(
        "Risk overview",
        "Financial risk assessment and prioritization"
    )

    df = current_matrix()
    completed = int(df.notna().sum().sum())

    c1, c2, c3, c4 = st.columns(4, gap="medium")

    c1.metric("Total risks", "05")
    c2.metric("Criteria", "04")
    c3.metric("MCDM methods", "09")
    c4.metric("Completion", f"{completed * 5}%")

    st.markdown(
        '<div class="section-title">Risk portfolio</div>',
        unsafe_allow_html=True
    )

    descriptions = [
        "Potential losses caused by counterparty default.",
        "Exposure to market price movements.",
        "Difficulty meeting payment obligations.",
        "Process, people or system failures.",
        "Cyberattacks and security incidents."
    ]

    symbols = ["◈", "↗", "◉", "⚙", "◇"]

    rows = []

    for i, risk in enumerate(RISKS):

        assessed = bool(df.loc[risk].notna().all())

        status = "Assessed" if assessed else "Pending"
        status_class = "complete" if assessed else ""

        rows.append(
            '<div class="portfolio-row">'
            f'<div class="portfolio-icon">{symbols[i]}</div>'
            '<div class="portfolio-text">'
            f'<div class="portfolio-name">{risk}</div>'
            '<div class="portfolio-description">'
            f'{descriptions[i]}'
            '</div></div>'
            f'<div class="portfolio-status {status_class}">'
            f'{status}</div>'
            '</div>'
        )

    st.markdown(
        '<div class="portfolio">'
        + "".join(rows)
        + '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-title">Evaluation framework</div>',
        unsafe_allow_html=True
    )

    
    st.markdown(
        '<div class="section-title">Evaluation framework</div>',
        unsafe_allow_html=True
    )

    framework = pd.DataFrame({
        "Criterion": [
            "Probability",
            "Financial impact",
            "Detection difficulty",
            "Recovery time"
        ],
        "Scale": ["1–9"] * 4,
        "Score of 9 means": [
            "Very high probability of occurrence",
            "Very severe potential financial loss",
            "Extremely difficult to detect",
            "Extremely long recovery time"
        ],
        "Direction": ["Maximize"] * 4
    })

    st.dataframe(
        framework,
        hide_index=True,
        use_container_width=True
    )


# ============================================================
# RISK ASSESSMENT
# ============================================================

elif page == "Risk Assessment":

    header(
        "Risk assessment",
        "Evaluate the five financial risks against four criteria."
    )

    st.info(
        "Enter a score from 1 to 10. "
        "A higher value indicates greater risk priority."
    )

    edited = st.data_editor(
        current_matrix(),
        key="risk_editor",
        use_container_width=True,
        num_rows="fixed",
        column_config={
            criterion: st.column_config.NumberColumn(
                criterion,
                min_value=1,
                max_value=9,
                step=1
            )
            for criterion in CRITERIA
        }
    )

    if st.button("Save assessment", type="primary"):

        try:
            x = edited.to_numpy(dtype=float)

            if not np.isfinite(x).all():
                raise ValueError("Complete every cell.")

            if (x < 1).any() or (x > 9).any():
                raise ValueError("Use scores between 1 and 9.")

            if not edited.equals(st.session_state.matrix):
                st.session_state.weights = {}

            st.session_state.matrix = edited.copy()
            st.success("Assessment saved successfully.")

        except (ValueError, TypeError) as error:
            st.error(str(error))

    st.download_button(
        "Export assessment",
        current_matrix().to_csv().encode("utf-8"),
        "risk_assessment.csv",
        "text/csv"
    )


# ============================================================
# CRITERIA WEIGHTING
# ============================================================

elif page == "Criteria Weighting":

    header(
        "Criteria weighting",
        "Calculate the relative importance of the evaluation criteria."
    )

    method = st.selectbox(
        "Select weighting method",
        ["AHP", "BWM", "Entropy", "CRITIC"]
    )

    weights = None
    detail = None

    if method == "AHP":

        st.write(
            "Compare each pair of criteria using Saaty's "
            "1–9 importance scale."
        )

        pairwise = np.ones((4, 4))

        for i in range(4):
            for j in range(i + 1, 4):

                c1, c2 = st.columns([3, 1])

                preference = c1.selectbox(
                    f"{CRITERIA[i]} vs {CRITERIA[j]}",
                    [
                        "Equal importance",
                        CRITERIA[i],
                        CRITERIA[j]
                    ],
                    key=f"preference_{i}_{j}"
                )

                intensity = c2.slider(
                    "Intensity",
                    1, 9, 1,
                    key=f"intensity_{i}_{j}"
                )

                if preference == CRITERIA[i]:
                    pairwise[i, j] = intensity
                elif preference == CRITERIA[j]:
                    pairwise[i, j] = 1 / intensity

                pairwise[j, i] = 1 / pairwise[i, j]

        with st.expander("View pairwise matrix"):
            st.dataframe(
                pd.DataFrame(
                    pairwise,
                    index=CRITERIA,
                    columns=CRITERIA
                ).round(3)
            )

        if st.button("Calculate AHP", type="primary"):
            weights, detail = ahp(pairwise)

            if detail >= 0.1:
                st.warning(
                    f"Consistency ratio: {detail:.3f}. "
                    "Revise the pairwise comparisons."
                )
            else:
                save_weights("AHP", weights)
                st.success(f"AHP saved. CR = {detail:.3f}")

    elif method == "BWM":

        best = st.selectbox(
            "Most important criterion",
            CRITERIA
        )

        worst = st.selectbox(
            "Least important criterion",
            CRITERIA,
            index=3
        )

        if best == worst:
            st.warning("Choose different best and worst criteria.")

        else:
            st.subheader("Best to Others")

            bo = [
                1 if criterion == best else st.slider(
                    f"{best} over {criterion}",
                    1, 9, 3,
                    key=f"bo_{criterion}"
                )
                for criterion in CRITERIA
            ]

            st.subheader("Others to Worst")

            ow = [
                1 if criterion == worst else st.slider(
                    f"{criterion} over {worst}",
                    1, 9, 3,
                    key=f"ow_{criterion}"
                )
                for criterion in CRITERIA
            ]

            if st.button("Calculate BWM", type="primary"):
                weights, detail = bwm(
                    CRITERIA.index(best),
                    CRITERIA.index(worst),
                    bo, ow
                )

                save_weights("BWM", weights)
                st.success(
                    f"BWM saved. Optimization deviation: "
                    f"{detail:.4f}"
                )

    else:

        try:
            x = get_matrix()

            if st.button(
                f"Calculate {method}",
                type="primary"
            ):

                if method == "Entropy":
                    weights, detail = entropy(x)
                else:
                    weights, detail = critic(x)

                save_weights(
                    method, weights,
                    data_dependent=True
                )

                st.success(f"{method} weights saved.")

                if method == "CRITIC":
                    st.dataframe(
                        pd.DataFrame(
                            detail,
                            index=CRITERIA,
                            columns=CRITERIA
                        ).round(3)
                    )

        except ValueError as error:
            st.warning(str(error))

    if weights is not None:

        st.markdown(
            '<div class="section-title">Calculated weights</div>',
            unsafe_allow_html=True
        )

        weight_df = pd.DataFrame({
            "Criterion": CRITERIA,
            "Weight": weights
        })

        st.dataframe(
            weight_df.style.format({"Weight": "{:.4f}"}),
            hide_index=True,
            use_container_width=True
        )

        fig = px.bar(
            weight_df,
            x="Criterion",
            y="Weight",
            color_discrete_sequence=["#168B77"]
        )

        st.plotly_chart(fig, use_container_width=True)


# ============================================================
# RISK RANKING
# ============================================================

elif page == "Risk Ranking":

    header(
        "Risk ranking",
        "Prioritize financial risks using MCDM ranking methods."
    )

    if not st.session_state.weights:
        st.warning("Calculate criteria weights first.")
        st.stop()

    try:
        x = get_matrix()

        weighting = st.selectbox(
            "Weighting method",
            list(st.session_state.weights)
        )

        weights = get_weights(weighting)

        method = st.selectbox(
            "Ranking method",
            ["WSM", "WPM", "WASPAS", "TOPSIS", "VIKOR"]
        )

        lam = 0.5
        v = 0.5

        if method == "WASPAS":
            lam = st.slider("WASPAS λ", 0.0, 1.0, 0.5)

        if method == "VIKOR":
            v = st.slider("VIKOR v", 0.0, 1.0, 0.5)

        result = ranking_table(
            x, weights, method, RISKS, lam, v
        )

        st.markdown(
            '<div class="section-title">Priority ranking</div>',
            unsafe_allow_html=True
        )

        st.dataframe(
            result,
            hide_index=True,
            use_container_width=True
        )

        fig = px.bar(
            result,
            x="Score",
            y="Risk",
            orientation="h",
            color_discrete_sequence=["#168B77"],
            text="Rank"
        )

        fig.update_layout(
            yaxis={
                "categoryorder": "array",
                "categoryarray": result["Risk"].iloc[::-1].tolist()
            },
            plot_bgcolor="white",
            paper_bgcolor="white"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.caption(
            "Rank 1 represents the highest modeled "
            "intervention priority. For VIKOR, lower Q "
            "values receive earlier positions."
        )

        if method == "VIKOR":
            ordered = result.sort_values("Score")
            advantage = (
                ordered["Score"].iloc[1]
                - ordered["Score"].iloc[0]
                >= 1 / (len(RISKS) - 1)
            )

            first = ordered["Risk"].iloc[0]
            stable = first in (
                result.loc[result["S"].idxmin(), "Risk"],
                result.loc[result["R"].idxmin(), "Risk"]
            )

            st.info(
                f"VIKOR compromise conditions — "
                f"acceptable advantage: {advantage}; "
                f"acceptable stability: {stable}."
            )

        st.download_button(
            "Export ranking",
            result.to_csv(index=False).encode(),
            "risk_ranking.csv",
            "text/csv"
        )

    except ValueError as error:
        st.warning(str(error))


# ============================================================
# COMPARISON & SENSITIVITY
# ============================================================

elif page == "Comparison & Sensitivity":

    header(
        "Comparison & sensitivity",
        "Compare ranking methods and explore changes in weights."
    )

    if not st.session_state.weights:
        st.warning("Calculate criteria weights first.")
        st.stop()

    try:
        x = get_matrix()

        weighting = st.selectbox(
            "Weighting method",
            list(st.session_state.weights)
        )

        weights = get_weights(weighting)

        methods = ["WSM", "WPM", "WASPAS", "TOPSIS", "VIKOR"]

        ranks = {}

        for method in methods:
            table = ranking_table(
                x, weights, method, RISKS
            )

            ranks[method] = (
                table.set_index("Risk")["Rank"]
            )

        comparison = pd.DataFrame(ranks).loc[RISKS]

        st.subheader("Ranking comparison")

        st.dataframe(
            comparison,
            use_container_width=True
        )

        fig = px.imshow(
            comparison,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="Tealgrn"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Spearman correlations")

        st.dataframe(
            comparison.corr(method="spearman").round(3),
            use_container_width=True
        )

        st.subheader("Sensitivity analysis")

        adjusted = np.array([
            st.slider(
                criterion,
                0.0, 1.0,
                float(weights[i]),
                0.01,
                key=f"sensitivity_{weighting}_{i}"
            )
            for i, criterion in enumerate(CRITERIA)
        ])

        if adjusted.sum() == 0:
            st.error("At least one weight must be positive.")
            st.stop()

        adjusted /= adjusted.sum()

        selected_method = st.selectbox(
            "Method for sensitivity",
            methods
        )

        original = ranking_table(
            x, weights, selected_method, RISKS
        )

        modified = ranking_table(
            x, adjusted, selected_method, RISKS
        )

        sensitivity = (
            original[["Risk", "Rank"]]
            .rename(columns={"Rank": "Original rank"})
            .merge(
                modified[["Risk", "Rank"]].rename(
                    columns={"Rank": "Adjusted rank"}
                ),
                on="Risk"
            )
        )

        st.dataframe(
            sensitivity,
            hide_index=True,
            use_container_width=True
        )

        rho = spearmanr(
            sensitivity["Original rank"],
            sensitivity["Adjusted rank"]
        ).statistic

        st.metric("Spearman correlation", f"{rho:.3f}")

        st.download_button(
            "Export comparison",
            comparison.to_csv().encode(),
            "ranking_comparison.csv",
            "text/csv"
        )

    except ValueError as error:
        st.warning(str(error))
