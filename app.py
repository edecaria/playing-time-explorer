import os
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# --- Define styles

st.markdown(
    """
    <style>
    .pill {
      display:inline-block;
      padding:2px 10px;
      border-radius:999px;
      font-size:0.75rem;
      font-weight:700;
      letter-spacing:0.04em;
      text-transform:uppercase;
    }
    .block-container {
        padding-top: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Define benchmarks

TIERS = [
    {"name": "ELITE", "bg": "#dcfce7", "fg": "#166534"},
    {"name": "GREAT", "bg": "#e0f2fe", "fg": "#075985"},
    {"name": "GOOD", "bg": "#fef9c3", "fg": "#854d0e"},
    {"name": "OKAY", "bg": "#ffedd5", "fg": "#9a3412"},
    {"name": "POOR", "bg": "#fee2e2", "fg": "#991b1b"},
]

BENCHMARKS = {
    "AW": {
        "thresholds": [27, 24, 19, 14],
        "display_range": (1.0, 27.0)
    },
    "PAAW": {
        "thresholds": [24.0, 22.0, 18.0, 14.0],
        "display_range": (1.0, 28.0)
    },
    "SAW": {
        "thresholds": [5.5, 4.9, 4.0, 3.0],
        "display_range": (2.0, 6.0)
    },
    "PAS": {
        "thresholds": [4.35, 4.20, 4.00, 3.80],
        "display_range": (3.5, 5.0)
    }}

column_formats = {
    "Total PA": st.column_config.NumberColumn(format="%d"),
    "Age": st.column_config.NumberColumn(format="%d"),
    "AW": st.column_config.NumberColumn(format="%d"),
    "PAAW": st.column_config.NumberColumn(format="%.1f"),
    "SAW": st.column_config.NumberColumn(format="%.1f"),
    "PAS": st.column_config.NumberColumn(format="%.2f"),
    "EAW": st.column_config.NumberColumn(format="%.1f"),
    "GSvR_pct": st.column_config.NumberColumn(label="%vR",format="%.0f%%"),
    "GSvL_pct": st.column_config.NumberColumn(label="%vL",format="%.0f%%"),
    "vR": st.column_config.NumberColumn(format="%.2f"),
    "vL": st.column_config.NumberColumn(format="%.2f"),
    "#R": st.column_config.NumberColumn(format="%d"),
    "#L": st.column_config.NumberColumn(format="%d"),
}

# --- Load data

@st.cache_data
def load_data():
    # Choose source (in priority order)
    if os.path.exists("mlb-playing-time.csv"):
        path = "mlb-playing-time.csv"
        df = pd.read_csv(path)

    else:
        data_url = st.secrets.get("AppDataURL", None)  # set this in your deployment environment
        if data_url:
            df = pd.read_csv(data_url)
        else:
            df = pd.read_csv("mlb-playing-time-sample.csv")

    # Normalize types (runs no matter which source was used)
    df["GSvR"] = pd.to_numeric(df["GSvR"], errors="coerce")
    df["GSvL"] = pd.to_numeric(df["GSvL"], errors="coerce")
    df["GSvR_pct"] = df["GSvR"] * 100
    df["GSvL_pct"] = df["GSvL"] * 100
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
    return df

df = load_data()

# Define parameters

def keep_existing(cols, df):
    """Return only the columns from cols that actually exist in df."""
    return [c for c in cols if c in df.columns]

def tier_for(metric: str, value: float):
    cfg = BENCHMARKS.get(metric)
    if cfg is None or value is None:
        return None

    thresholds = cfg["thresholds"]
    # thresholds correspond to TIERS[0:-1]
    for i, t in enumerate(thresholds):
        if value >= t:
            return i  # 0..len(thresholds)-1
    return len(TIERS) - 1  # POOR

def tier_style(tier_idx: int):
    return TIERS[tier_idx]["name"], TIERS[tier_idx]["bg"], TIERS[tier_idx]["fg"]

def tier_range(metric: str, tier_idx: int, col_min: float, col_max: float):
    """Return (min_val, max_val) for the given metric and tier index."""
    cfg = BENCHMARKS.get(metric)
    if cfg is None:
        return (col_min, col_max)

    thresholds = cfg["thresholds"]

    if tier_idx == 0:  # EXCELLENT: >= threshold[0]
        return (float(thresholds[0]), col_max)
    elif tier_idx < len(thresholds):  # GREAT, GOOD, OKAY
        return (float(thresholds[tier_idx]), float(thresholds[tier_idx - 1]))
    else:  # POOR: < threshold[-1]
        return (col_min, float(thresholds[-1]))

def sparkline(series, x=None):
    """Return a tiny Plotly figure for a sparkline."""
    if x is None:
        x = list(range(len(series)))

    fig = go.Figure(go.Scatter(x=x, y=series, mode="lines+markers"))
    fig.update_layout(
        height=90,
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        showlegend=False,
    )
    return fig

def tier_for(metric: str, value: float):
    cfg = BENCHMARKS.get(metric)
    if cfg is None or value is None:
        return None

    thresholds = cfg["thresholds"]  # length = len(TIERS)-1
    for i, t in enumerate(thresholds):
        if value >= t:
            return i
    return len(TIERS) - 1  # last tier (POOR)

st.set_page_config(page_title="Playing Time Explorer", layout="wide", initial_sidebar_state="collapsed")
st.title("Playing Time Explorer")

# Guardrails
required = {"Name", "Year"}
missing = required - set(df.columns)
if missing:
    st.error(f"Missing required columns: {sorted(missing)}")
    st.stop()

# --- Set mode (Player vs. League)

# Initialize session state for selected player
if "selected_player" not in st.session_state:
    st.session_state.selected_player = None
if "jump_to_player" not in st.session_state:
    st.session_state.jump_to_player = False

# Check query params for player navigation
query_params = st.query_params
if "player" in query_params:
    st.session_state.selected_player = query_params["player"]
    st.session_state.jump_to_player = True
    st.query_params.clear()
    st.rerun()

# Determine mode
mode_options = ["League", "Player"]
scroll_to_top = st.session_state.jump_to_player

# Force switch to Player mode when jumping from League table
if st.session_state.jump_to_player:
    st.session_state.mode = "Player"
    st.session_state.jump_to_player = False

if "mode" not in st.session_state:
    st.session_state.mode = mode_options[0]

mode = st.segmented_control(
    "View mode",
    mode_options,
    key="mode",
    label_visibility="collapsed"
)

# Scroll to top after jump
if scroll_to_top:
    st.components.v1.html("""
        <script>
            window.parent.scrollTo(0, 0);
            window.parent.document.body.scrollTo(0, 0);
            var main = window.parent.document.querySelector('[data-testid="stMain"]');
            if (main) main.scrollTo(0, 0);
            var container = window.parent.document.querySelector('[data-testid="stAppViewContainer"]');
            if (container) container.scrollTo(0, 0);
        </script>
    """, height=0)

if mode == "League":
    season = st.selectbox(
        "Season",
        sorted(df["Year"].dropna().unique()),
        index=len(df["Year"].dropna().unique()) - 1  # default = latest year
    )
    team = st.selectbox(
        "Team",
        ["All"] + sorted(df["Tm"].dropna().unique()),
        index=0
    )
    pos = st.selectbox(
        "Position",
        ["All", "2", "3", "4", "5", "6", "o", "0"],
        index=0
    )


    season_df = df[(df["Year"] == season) & ((team == "All") |(df["Tm"] == team)) & ((pos == "All") | (df["Eligible"].str.contains(pos)))].copy()
    
    fig = px.scatter(
        season_df,
        x="AW",
        y="PAAW",
        hover_name="Name",
        custom_data=["Name"],
        title=None,
    )

    # Light polish
    fig.update_layout(
        xaxis_title="Active Weeks (AW)",
        yaxis_title="Plate Appearances per Active Week (PAAW)",
        margin=dict(l=10, r=10, t=10, b=10),
    )

    chart_selection = st.plotly_chart(fig, on_select="rerun", selection_mode="points")

    # Handle chart click - jump to Player view
    if chart_selection and chart_selection.selection.points:
        clicked_name = chart_selection.selection.points[0]["customdata"][0]
        st.session_state.selected_player = clicked_name
        st.session_state.jump_to_player = True
        st.rerun()

    # --- Metric range filters (>= and <=) for AW, PAAW, SAW, PAS
    filter_metrics = [m for m in ["AW", "PAAW", "SAW", "PAS"] if m in season_df.columns and season_df[m].dropna().any()]
    metric_ranges = {}

    # Tier shortcut callback
    def on_tier_select(metric, c_min, c_max, is_int):
        selected = st.session_state[f"tier_{metric}"]
        if selected == "All":
            if is_int:
                st.session_state[f"filter_{metric}"] = (int(c_min), int(c_max))
            else:
                st.session_state[f"filter_{metric}"] = (c_min, c_max)
        else:
            tier_idx = next(i for i, t in enumerate(TIERS) if t["name"] == selected)
            lo, hi = tier_range(metric, tier_idx, c_min, c_max)
            if is_int:
                st.session_state[f"filter_{metric}"] = (int(lo), int(hi))
            else:
                st.session_state[f"filter_{metric}"] = (lo, hi)

    if filter_metrics:
        with st.expander("Filters", icon=":material/filter_list:"):
            # arrange two metrics per row
            cols_per_row = 2
            cols = st.columns(cols_per_row)
            for i, m in enumerate(filter_metrics):
                col = cols[i % cols_per_row]
                with col:
                    col_min = float(season_df[m].min())
                    col_max = float(season_df[m].max())
                    is_int = pd.api.types.is_integer_dtype(season_df[m])

                    # Initialize session state for this filter if not set
                    if f"filter_{m}" not in st.session_state:
                        if is_int:
                            st.session_state[f"filter_{m}"] = (int(col_min), int(col_max))
                        else:
                            st.session_state[f"filter_{m}"] = (col_min, col_max)

                    # Tier shortcut for AW - put tier and slider on same line
                    if m == "AW":
                        tier_options = ["All"] + [t["name"] for t in TIERS]
                        subcol1, subcol2 = st.columns([1, 2])
                        with subcol1:
                            st.selectbox(
                                m,
                                tier_options,
                                key=f"tier_{m}",
                                on_change=on_tier_select,
                                args=(m, col_min, col_max, is_int)
                            )
                        with subcol2:
                            min_v = int(col_min)
                            max_v = int(col_max)
                            val = st.slider(m, min_value=min_v, max_value=max_v, step=1, key=f"filter_{m}", label_visibility="hidden")
                            metric_ranges[m] = (val[0], val[1])
                        continue

                    # Tier shortcut for PAAW, SAW, PAS - put tier and slider on same line
                    if m in ["PAAW", "SAW", "PAS"]:
                        tier_options = ["All"] + [t["name"] for t in TIERS]
                        subcol1, subcol2 = st.columns([1, 2])
                        with subcol1:
                            st.selectbox(
                                m,
                                tier_options,
                                key=f"tier_{m}",
                                on_change=on_tier_select,
                                args=(m, col_min, col_max, is_int)
                            )
                        with subcol2:
                            step = (col_max - col_min) / 100 if col_max > col_min else 0.1
                            if step == 0:
                                step = 0.1
                            val = st.slider(m, min_value=col_min, max_value=col_max, step=0.1, format="%.1f", key=f"filter_{m}", label_visibility="hidden")
                            metric_ranges[m] = (float(val[0]), float(val[1]))
                        continue

                    # if is_int:
                    #     step = 1
                    #     min_v = int(col_min)
                    #     max_v = int(col_max)
                    #     val = st.slider(f"{m}", min_value=min_v, max_value=max_v, step=step, key=f"filter_{m}")
                    #     metric_ranges[m] = (val[0], val[1])
                    # else:
                    #     # float slider with reasonable step
                    #     step = (col_max - col_min) / 100 if col_max > col_min else 0.1
                    #     if step == 0:
                    #         step = 0.1
                    #     val = st.slider(f"{m}", min_value=col_min, max_value=col_max, step=step, format="%.2f", key=f"filter_{m}")
                    #     metric_ranges[m] = (float(val[0]), float(val[1]))

        # apply ranges (outside expander so filtering still works when collapsed)
        for m, (lo, hi) in metric_ranges.items():
            season_df = season_df[season_df[m].notna()]
            season_df = season_df[(season_df[m] >= lo) & (season_df[m] <= hi)]

    st.subheader("Metrics by season")
    league_cols = keep_existing(
        ["Year", "IDfg", "Name", "Age", "Hand", "Tm", "Primary", "Eligible",
         "Total PA", "AW", "PAAW", "SAW", "PAS", "EAW",
         "GSvR_pct", "GSvL_pct", "vR", "vL", "#R", "#L"],
        season_df
    )

    display_df = season_df[league_cols].reset_index(drop=True)

    selection = st.dataframe(
        display_df,
        height=600,
        on_select="rerun",
        selection_mode="single-row",
        column_config=column_formats,
        hide_index=True,
    )

    # Handle row selection - jump to Player view
    if selection and selection.selection.rows:
        selected_idx = selection.selection.rows[0]
        selected_name = display_df.iloc[selected_idx]["Name"]
        st.session_state.selected_player = selected_name
        st.session_state.jump_to_player = True
        st.rerun()


if mode == "Player":
    names = sorted(df["Name"].dropna().unique())
    # Use selected player from League view if available, otherwise default
    if st.session_state.selected_player and st.session_state.selected_player in names:
        default_name = st.session_state.selected_player
        st.session_state.selected_player = None  # Clear after use
    else:
        default_name = "Soto, Juan"
    default_index = names.index(default_name) if default_name in names else 0
    player = st.selectbox("Player", names, index=default_index, placeholder="Type to search…", key="player")
    years = sorted(df.loc[df["Name"] == player, "Year"].dropna().unique())

    d = df[(df["Name"] == player) & (df["Year"].isin(years))].copy()
    d = d.sort_values("Year")

    d["AW"] = d["AW"].astype(float)
    mask = d["Year"] == 2020
    d.loc[mask, "AW"] = d.loc[mask, "AW"] * 2.7
    
    # --- Sparklines

    st.subheader(player)

    key_metrics = [m for m in ["AW", "PAAW", "SAW", "PAS"] if m in d.columns]

    if not key_metrics:
        st.info("No key metrics found (AW/PAAW/SAW/PAS/EAW).")
    else:
        cols = st.columns(len(key_metrics))

        # Ensure consistent ordering by year
        d2 = d.sort_values("Year")
        years = d2["Year"].tolist()

        for col, m in zip(cols, key_metrics):
            with col:
                # "Card"
                with st.container(border=True):
                    latest = d2[m].dropna().iloc[-1] if d2[m].notna().any() else None
                    st.caption(m)
                    idx = tier_for(m, float(latest))
                    if idx is not None:
                        name, bg, fg = tier_style(idx)
                        st.markdown(
                            f"<span class='pill' style='background:{bg}; color:{fg};'>{name}</span>",
                            unsafe_allow_html=True,
                        )
                    if latest is None:
                        st.write("—")
                    else:
                        # format: ints vs floats
                        if pd.api.types.is_numeric_dtype(d2[m]):
                            st.markdown(f"### {latest:.2f}")
                        else:
                            st.markdown(f"### {latest}")

                    # sparkline
                    s = d2[m]
                    # keep only numeric for charting
                    if pd.api.types.is_numeric_dtype(s):
                        fig = sparkline(s.tolist(), x=years)
                        st.plotly_chart(fig, config={"displayModeBar": False})

    # --- Tabs

    tab1, tab2 = st.tabs(
        ["Building Blocks", "Handedness Splits"]
    )

    with tab1:
        cols = keep_existing(["Total PA", "Age", "AW", "PAAW", "SAW", "PAS", "EAW"], d)
        st.dataframe(d[["Year"] + cols], column_config=column_formats, hide_index=True)
    with tab2:
        cols = keep_existing(["Age", "GSvR_pct", "GSvL_pct", "vR", "vL", "#R", "#L"], d)
        st.dataframe(d[["Year"] + cols], column_config=column_formats, hide_index=True)
#    with tab3:
#        cols = keep_existing(["AB/AW", "H/AW", "R/AW", "HR/AW", "RBI/AW", "SB/AW"], d)
#        st.dataframe(d[["Year"] + cols], column_config=column_formats)
