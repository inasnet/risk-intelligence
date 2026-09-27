
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.stats import spearmanr
from mcdm_methods import (
    ahp, bwm, entropy, critic, ranking_table
)


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


st.markdown("<style>" + (Path(__file__).parent / "styles.css").read_text(encoding="utf-8") + "</style>", unsafe_allow_html=True)


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
        ("dashboard", "Overview", "Overview"),
        ("fact_check", "Risk Assessment", "Risk Assessment"),
        ("tune", "Criteria Weighting", "Criteria Weighting"),
        ("leaderboard", "Risk Ranking", "Risk Ranking"),
        ("compare_arrows", "Comparison & Sensitivity", "Comparison")
    ]

    for icon, destination, label in NAVIGATION:
        active = st.session_state.page == destination
        if st.button(
            label,
            icon=f":material/{icon}:",
            help="Compare methods and explore sensitivity" if destination == "Comparison & Sensitivity" else None,
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
        '<div class="sidebar-progress-heading"><div class="sidebar-progress-title">'
        'Assessment progress</div>'
        f'<div class="sidebar-progress-value">{percentage}%</div></div>'
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

    st.markdown(
        '<div class="sidebar-footer"><span class="footer-mark">RI</span>'
        '<div>Risk Intelligence<small>MCDM Research Project</small></div></div>',
        unsafe_allow_html=True
    )


page = st.session_state.page



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
    c3.metric("MCDM methods", "08")
    c4.metric("Completion", f"{completed * 5}%")

    ready = completed == df.size
    assessed_count = int(df.notna().all(axis=1).sum())
    portfolio_col, guide_col = st.columns([2.1, 1], gap="large")

    icon_paths = [
        '<path d="m3 9 9-6 9 6M4 10h16M6 10v8m6-8v8m6-8v8M3 21h18"/>',
        '<path d="m3 17 6-6 4 4 8-10m-6 0h6v6"/>',
        '<rect x="3" y="6" width="18" height="15" rx="3"/><path d="M3 9V5a2 2 0 0 1 2-2h13m-2 10h5m-4 3h1"/>',
        '<path d="M4 21V9l5 3V7l6 3V3h5v18ZM8 17h1m4 0h1m3 0h1"/>',
        '<path d="m12 3 8 3v6c0 5-8 9-8 9s-8-4-8-9V6Z"/><path d="m8 12 3 3 5-6"/>'
    ]
    rows = []
    for i, risk in enumerate(RISKS):
        entered = int(df.loc[risk].notna().sum())
        status = "Assessed" if entered == 4 else ("In progress" if entered else "Not assessed")
        status_class = "complete" if entered == 4 else ("partial" if entered else "")
        icon = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">' + icon_paths[i] + '</svg>'
        rows.append(
            f'<div class="portfolio-row tone-{i}">'
            f'<div class="portfolio-icon">{icon}</div>'
            '<div class="portfolio-text">'
            f'<div class="portfolio-name">{risk}</div>'
            f'<div class="portfolio-description">{RISK_INFO[risk][1]}</div>'
            '</div><div class="risk-state">'
            f'<span class="portfolio-status {status_class}">{status}</span>'
            f'<span class="score-count">{entered} / 4 criteria</span>'
            '</div></div>'
        )
    with portfolio_col:
        st.markdown(
            '<div class="section-heading"><h2>Risk portfolio</h2>'
            f'<span>{assessed_count} of 5 assessed</span></div>'
            '<div class="portfolio">' + ''.join(rows) + '</div>',
            unsafe_allow_html=True
        )
    with guide_col:
        st.markdown('<div class="section-heading"><h2>Your next step</h2><span>WORKFLOW</span></div>', unsafe_allow_html=True)
        with st.container(border=True):
            pct = int(completed / df.size * 100)
            st.markdown(
                '<div class="guide-kicker">ASSESSMENT READINESS</div>'
                f'<div class="readiness"><div class="progress-ring" style="--progress:{pct}%">'
                f'<div>{pct}<small>%</small></div></div>'
                f'<div><strong>{completed} of 20</strong><span>scores completed</span></div></div>'
                '<div class="guide-title">'
                + ('Ready to set your priorities' if ready else 'Build your risk profile')
                + '</div><div class="guide-copy">'
                + ('Your scores are saved. Define how much each criterion matters before ranking.' if ready
                   else 'Evaluate each risk across four criteria to begin your analysis.')
                + '</div>', unsafe_allow_html=True
            )
            if st.button(
                "Set criteria weights" if ready else ("Continue assessment" if completed else "Start risk assessment"),
                key="overview_next_step", type="primary", use_container_width=True
            ):
                st.session_state.page = "Criteria Weighting" if ready else "Risk Assessment"
                st.rerun()
            st.markdown(
                '<div class="workflow-steps">'
                '<div class="step active"><b>01</b><span>Assess risks<small>Score your five risk categories</small></span></div>'
                '<div class="step"><b>02</b><span>Weight criteria<small>Define relative importance</small></span></div>'
                '<div class="step"><b>03</b><span>Compare rankings<small>Explore methods and sensitivity</small></span></div>'
                '</div>', unsafe_allow_html=True
            )

    with st.expander("Evaluation framework | Scoring guide", expanded=False):
        st.caption("Score each criterion from 1 to 9. Higher scores indicate greater risk priority.")
        framework = pd.DataFrame({
            "Criterion": CRITERIA,
            "Scale": ["1-9"] * 4,
            "Score of 9 means": [
                "Very high probability of occurrence",
                "Very severe potential financial loss",
                "Extremely difficult to detect",
                "Extremely long recovery time"
            ],
            "Direction": ["Maximize"] * 4
        })
        st.dataframe(framework, hide_index=True, use_container_width=True)



elif page == "Risk Assessment":

    header(
        "Risk assessment",
        "Evaluate the five financial risks against four criteria."
    )

    st.info(
        "Enter a score from 1 to 9. "
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
                    min_value=1,
                    max_value=9,
                    value=1,
                    disabled=(preference == "Equal importance"),
                    key=f"intensity_{i}_{j}"
                )

                if preference == CRITERIA[i]:
                    pairwise[i, j] = intensity

                elif preference == CRITERIA[j]:
                    pairwise[i, j] = 1 / intensity

                else:
                    pairwise[i, j] = 1

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

###3RISK RANKING 

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
            ["WSM", "WPM", "WASPAS", "TOPSIS"]
        )

        lam = 0.5

        if method == "WASPAS":
            lam = st.slider("WASPAS λ", 0.0, 1.0, 0.5)

        result = ranking_table(
            x, weights, method, RISKS, lam
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
            "intervention priority."
        )

        st.download_button(
            "Export ranking",
            result.to_csv(index=False).encode(),
            "risk_ranking.csv",
            "text/csv"
        )

    except ValueError as error:
        st.warning(str(error))


## COMPARAISON ET SENSTIVIT2 

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

        methods = ["WSM", "WPM", "WASPAS", "TOPSIS"]

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
