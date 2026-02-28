import os
import streamlit as st
import plotly.graph_objects as go
import polars as pl


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
    h3 {
        padding-bottom: 0 !important;
    }
    div[data-testid="stDataFrameToolbar"],
    div[data-testid="stDataFrameToolbar"] * ,
    button[data-testid="stDataFrameToolbarButton"],
    div[data-testid="stElementToolbar"] {
        display: none !important;
        visibility: hidden !important;
        pointer-events: none !important;
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
    {"name": "BORDERLINE", "bg": "#ffedd5", "fg": "#9a3412"},
    {"name": "BAD", "bg": "#fee2e2", "fg": "#991b1b"},
]

BENCHMARKS = {
    "AW": {
        "thresholds": [27, 24, 19, 14],
        "display_range": (0.0, 28.0)
    },
    "PAAW": {
        "thresholds": [24.0, 22.0, 18.0, 14.0],
        "display_range": (1.0, 30.0)
    },
    "SAW": {
        "thresholds": [5.5, 4.9, 4.0, 3.0],
        "display_range": (0.0, 6.1)
    },
    "PAS": {
        "thresholds": [4.35, 4.20, 4.00, 3.80],
        "display_range": (1.0, 5.0)
    }}

column_formats = {
    "Name": st.column_config.TextColumn(width=160),
    "Tm": st.column_config.TextColumn(width=40),
    "Team26": st.column_config.TextColumn(width=40),
    "Pos26": st.column_config.TextColumn(width=40),
    "Hand": st.column_config.TextColumn(width=40),
    "Bat": st.column_config.TextColumn(width=35),
    "Primary": st.column_config.TextColumn(width=40),
    "Pos": st.column_config.TextColumn(width=40),
    "Eligible": st.column_config.TextColumn(width=40),
    "Elig": st.column_config.TextColumn(width=40),
    "IDfg": st.column_config.NumberColumn(format="%d", width=80),
    "Year": st.column_config.NumberColumn(format="%d", width=45),
    "Total PA": st.column_config.NumberColumn(format="%d"),
    "PA": st.column_config.NumberColumn(format="%d",width=40),
    "Age": st.column_config.NumberColumn(format="%d", width=35),
    "AW": st.column_config.NumberColumn(format="%d", width=35),
    "PAAW": st.column_config.NumberColumn(format="%.1f", width=50),
    "SAW": st.column_config.NumberColumn(format="%.1f", width=40),
    "PAS": st.column_config.NumberColumn(format="%.2f", width=40),
    "EAW": st.column_config.NumberColumn(format="%.1f", width=40),
    "GSvR_pct": st.column_config.NumberColumn(label="%vR", format="%.0f%%",width=45),
    "GSvL_pct": st.column_config.NumberColumn(label="%vL", format="%.0f%%", width=45),
    "vR": st.column_config.NumberColumn(format="%.2f",width=40),
    "vL": st.column_config.NumberColumn(format="%.2f",width=40),
    "#R": st.column_config.NumberColumn(format="%d", width=30),
    "#L": st.column_config.NumberColumn(format="%d", width=30),
    "wOBA": st.column_config.NumberColumn(format="%.3f",width=50),
    "wOBA vR": st.column_config.NumberColumn(format="%.3f",width=60),
    "wOBA vL": st.column_config.NumberColumn(format="%.3f",width=60),
    "wTMx+": st.column_config.NumberColumn(format="%d",width=50),
    "wTmX+": st.column_config.NumberColumn(format="%d",width=50),
    "wTmX+R": st.column_config.NumberColumn(format="%d",width=70),
    "wTmX+L": st.column_config.NumberColumn(format="%d",width=70),
}

# --- Load data

@st.cache_data
def load_data():
    # Choose source (in priority order)
    # 1) URL Parquet (preferred)
    used_parquet = False
    data_url = st.secrets.get("AppDataURL", None)  # set this in your deployment environment
    if data_url:
        try:
            df = pl.read_parquet(data_url)
            used_parquet = True
        except Exception:
            used_parquet = False

    if not used_parquet:
        # 2) Local parquet (repo root)
        local_parquet = "batting_season_all.parquet"
        if os.path.exists(local_parquet):
            df = pl.read_parquet(local_parquet)
            used_parquet = True
        else:
            raise FileNotFoundError(
                "No parquet source found. Set AppDataURL in secrets or add "
                "batting_season_all.parquet to the repo root."
            )

    if used_parquet:
        # Normalize Parquet schema to the app's existing column names.
        rename_map = {
            "season": "Year",
            "age": "Age",
            "Bats": "Hand",
            "team": "Tm",
            "team_now": "Team26",
            "pos_primary": "Primary",
            "pos_eligible": "Eligible",
            "pa_total": "Total PA",
            "active_weeks": "AW",
            "paaw": "PAAW",
            "saw": "SAW",
            "pas": "PAS",
            "eaw": "EAW",
            "gs_pct_r": "GSvR",
            "gs_pct_l": "GSvL",
            "pas_r": "vR",
            "pas_l": "vL",
            "lineup_slot_mode_r": "#R",
            "lineup_slot_mode_l": "#L",
            "woba": "wOBA",
            "woba_rhp": "wOBA vR",
            "woba_lhp": "wOBA vL",
            "wtmx_plus": "wTmX+",
            "wtmx_plus_rhp": "wTmX+R",
            "wtmx_plus_lhp": "wTmX+L",
        }
        df = df.rename({k: v for k, v in rename_map.items() if k in df.columns})

        if "Name" not in df.columns and {"LastName", "FirstName"}.issubset(df.columns):
            df = df.with_columns(
                (
                    pl.col("LastName").fill_null("").cast(pl.Utf8)
                    + pl.lit(", ")
                    + pl.col("FirstName").fill_null("").cast(pl.Utf8)
                ).str.strip_chars().str.strip_chars(",").str.strip_chars().alias("Name")
            )
            df = df.with_columns(
                pl.when(pl.col("Name") == "").then(pl.lit(None)).otherwise(pl.col("Name")).alias("Name")
            )

        if "Pos26" not in df.columns and "Primary" in df.columns:
            df = df.with_columns(pl.col("Primary").alias("Pos26"))

#        if "IDfg" not in df.columns and "MLBAMId" in df.columns:
#            df = df.with_columns(pl.col("MLBAMId").alias("IDfg"))

    if "is_split" in df.columns:
        df = df.with_columns(pl.col("is_split").cast(pl.Int64, strict=False).fill_null(0).alias("IsSplit"))
    elif "IsSplit" not in df.columns:
        df = df.with_columns(pl.lit(0).alias("IsSplit"))

    # Normalize types (runs no matter which source was used)
    numeric_casts = []
    for col in ["GSvR", "GSvL", "vR", "vL", "#R", "#L", "AW", "PAAW", "SAW", "PAS", "EAW", "Age", "Total PA", "Year", "wOBA", "wTmX+", "wTmX+R", "wTmX+L"]:
        if col in df.columns:
            target = pl.Int64 if col in ["#R", "#L", "Age", "Total PA", "Year", "wTmX+", "wTmX+R", "wTmX+L"] else pl.Float64
            numeric_casts.append(pl.col(col).cast(target, strict=False).alias(col))
    if numeric_casts:
        df = df.with_columns(numeric_casts)

    derived = []
    if "GSvR" in df.columns:
        derived.append((pl.col("GSvR") * 100).alias("GSvR_pct"))
    if "GSvL" in df.columns:
        derived.append((pl.col("GSvL") * 100).alias("GSvL_pct"))
    if derived:
        df = df.with_columns(derived)

    return df

df = load_data()
df_regular = df.filter(pl.col("IsSplit") == 0) if "IsSplit" in df.columns else df

# Define parameters

def keep_existing(cols, df):
    """Return only the columns from cols that actually exist in df."""
    return [c for c in cols if c in df.columns]

def full_table_height(n_rows: int) -> int:
    """Return a dataframe height that fits all rows (no internal scroll)."""
    return 42 + max(1, n_rows) * 35

def sorted_unique_non_null(frame: pl.DataFrame, col: str):
    if col not in frame.columns:
        return []
    return sorted(frame.get_column(col).drop_nulls().unique().to_list())

def col_has_any_non_null(frame: pl.DataFrame, col: str) -> bool:
    if col not in frame.columns:
        return False
    return bool(frame.select(pl.col(col).is_not_null().any()).item())

def col_last_non_null(frame: pl.DataFrame, col: str):
    if col not in frame.columns:
        return None
    s = frame.get_column(col).drop_nulls()
    return s[-1] if len(s) > 0 else None

def compute_age_for_year(frame: pl.DataFrame, target_year: int) -> int | None:
    if not {"Year", "Age"}.issubset(frame.columns):
        return None
    tmp = frame.select(["Year", "Age"]).drop_nulls()
    if tmp.height == 0:
        return None
    tmp = tmp.sort("Year")
    year, age = tmp.row(-1)
    if year is None or age is None:
        return None
    return int(age + (target_year - int(year)))

def is_integer_dtype(dtype: pl.DataType) -> bool:
    return dtype in [pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64]

def is_numeric_dtype(dtype: pl.DataType) -> bool:
    return dtype in [
        pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
        pl.Float32, pl.Float64,
    ]

def filter_non_pitchers(frame: pl.DataFrame) -> pl.DataFrame:
    """Return frame filtered to non-pitchers based on position columns."""
    if "Primary" not in frame.columns:
        return frame

    primary = pl.col("Primary").fill_null("").cast(pl.Utf8, strict=False).str.to_uppercase()
    # Treat any Primary that includes P / SP / RP or equals numeric 1 as pitcher
    non_pitcher = (
        (primary != "")
    )
    return frame.filter(non_pitcher)

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

def sparkline(series, x=None, y_range=None):
    """Return a tiny Plotly figure for a sparkline."""
    if x is None:
        x = list(range(len(series)))

    fig = go.Figure(go.Scatter(x=x, y=series, mode="lines+markers", line=dict(color="#27245C"), marker=dict(color="#27245C")))

    yaxis_config = dict(visible=False)
    if y_range:
        yaxis_config["range"] = [y_range[0], y_range[1]]

    fig.update_layout(
        height=90,
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(visible=False),
        yaxis=yaxis_config,
        showlegend=False,
    )
    return fig

# def tier_for(metric: str, value: float):
#     cfg = BENCHMARKS.get(metric)
#     if cfg is None or value is None:
#         return None
#
#     thresholds = cfg["thresholds"]  # length = len(TIERS)-1
#     for i, t in enumerate(thresholds):
#         if value >= t:
#             return i
#     return len(TIERS) - 1  # last tier (POOR)

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

# Initialize filter session states (so they persist across tab switches)
if "filter_season" not in st.session_state:
    season_values = sorted_unique_non_null(df_regular, "Year")
    st.session_state.filter_season = max(season_values) if season_values else None
if "filter_team" not in st.session_state:
    st.session_state.filter_team = "All"
if "filter_team26" not in st.session_state:
    st.session_state.filter_team26 = "All"
if "filter_pos" not in st.session_state:
    st.session_state.filter_pos = "All"
if "filter_pos26" not in st.session_state:
    st.session_state.filter_pos26 = "All"
if "show_partial_seasons" not in st.session_state:
    st.session_state.show_partial_seasons = False

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
    years_list = sorted_unique_non_null(df_regular, "Year")
    years_list = sorted(years_list, reverse=True)
    season_idx = years_list.index(st.session_state.filter_season) if st.session_state.filter_season in years_list else 0
    season = st.selectbox("Season", years_list, index=season_idx)
    st.session_state.filter_season = season

    # Get season-only filtered df for computing metric slider ranges
    season_only_df = df_regular.filter(pl.col("Year") == season)

    # --- Metric range filters setup
    filter_metrics = [m for m in ["AW", "PAAW", "SAW", "PAS"] if m in season_only_df.columns and col_has_any_non_null(season_only_df, m)]
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

    with st.expander("Filters", icon=":material/filter_list:"):
        # Team filters
        team_values = [
            t for t in sorted_unique_non_null(df_regular, "Tm")
            if not (isinstance(t, str) and len(t) == 3 and t[0].isdigit() and t[1:].upper() == "TM")
        ]
        team_options = ["All"] + team_values
        team26_options = ["All"] + sorted_unique_non_null(df_regular, "Team26")
        pos_options = ["All", "2", "3", "4", "5", "6", "o", "0"]
        pos26_options = ["All", "C", "1B", "2B", "3B", "SS", "OF", "UT"]

        team_col1, team_col2 = st.columns(2)
        with team_col1:
            team_idx = team_options.index(st.session_state.filter_team) if st.session_state.filter_team in team_options else 0
            team = st.selectbox("Team", team_options, index=team_idx)
            st.session_state.filter_team = team
        with team_col2:
            team26_idx = team26_options.index(st.session_state.filter_team26) if st.session_state.filter_team26 in team26_options else 0
            team26 = st.selectbox("2026 Team", team26_options, index=team26_idx)
            st.session_state.filter_team26 = team26
        # Position filters
        pos_col1, pos_col2 = st.columns(2)
        with pos_col1:
            pos_idx = pos_options.index(st.session_state.filter_pos) if st.session_state.filter_pos in pos_options else 0
            pos = st.selectbox("Eligibility", pos_options, index=pos_idx)
            st.session_state.filter_pos = pos
        with pos_col2:
            pos26_idx = pos26_options.index(st.session_state.filter_pos26) if st.session_state.filter_pos26 in pos26_options else 0
            pos26 = st.selectbox("2026 Position", pos26_options, index=pos26_idx)
            st.session_state.filter_pos26 = pos26
        # Metric range filters
        if filter_metrics:
            cols_per_row = 2
            cols = st.columns(cols_per_row)
            for i, m in enumerate(filter_metrics):
                col = cols[i % cols_per_row]
                with col:
                    col_min = float(season_only_df.select(pl.col(m).min()).item())
                    col_max = float(season_only_df.select(pl.col(m).max()).item())
                    is_int = is_integer_dtype(season_only_df.schema.get(m))

                    # Initialize session state for this filter if not set
                    if f"filter_{m}" not in st.session_state:
                        if is_int:
                            st.session_state[f"filter_{m}"] = (int(col_min), int(col_max))
                        else:
                            st.session_state[f"filter_{m}"] = (col_min, col_max)

                    # Tier shortcut for AW
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
                            current_val = st.session_state.get(f"filter_{m}", (min_v, max_v))
                            # Clamp to valid range
                            current_val = (max(min_v, int(current_val[0])), min(max_v, int(current_val[1])))
                            val = st.slider(m, min_value=min_v, max_value=max_v, value=current_val, step=1, label_visibility="hidden")
                            st.session_state[f"filter_{m}"] = val
                            metric_ranges[m] = (val[0], val[1])
                        continue

                    # Tier shortcut for PAAW, SAW, PAS
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
                            step = 0.1
                            current_val = st.session_state.get(f"filter_{m}", (col_min, col_max))
                            # Clamp to valid range
                            current_val = (max(col_min, float(current_val[0])), min(col_max, float(current_val[1])))
                            val = st.slider(m, min_value=col_min, max_value=col_max, value=current_val, step=step, format="%.1f", label_visibility="hidden")
                            st.session_state[f"filter_{m}"] = val
                            metric_ranges[m] = (float(val[0]), float(val[1]))
                        continue

    # Apply all filters
    base_df = df if team != "All" else df_regular
    season_df = base_df.filter(
        (pl.col("Year") == season)
        & (pl.lit(team == "All") | (pl.col("Tm") == team))
        & (pl.lit(pos == "All") | pl.col("Eligible").fill_null("").cast(pl.Utf8).str.contains(pos, literal=True))
        & (pl.lit(team26 == "All") | (pl.col("Team26") == team26))
        & (pl.lit(pos26 == "All") | pl.col("Pos26").fill_null("").cast(pl.Utf8).str.contains(pos26, literal=True))
    )

    # Remove rows without a primary position
    if "Primary" in season_df.columns:
        season_df = season_df.filter(
            pl.col("Primary").is_not_null()
            & (pl.col("Primary").cast(pl.Utf8).str.strip_chars() != "")
        )

    # Apply metric range filters
    for m, (lo, hi) in metric_ranges.items():
        season_df = season_df.filter(pl.col(m).is_not_null() & pl.col(m).is_between(lo, hi, closed="both"))

    # st.caption(f"Players matching filters: {season_df.height}")

    # Charts (collapsed by default)
    with st.expander("Charts", icon=":material/monitoring:", expanded=False):
        with st.container():
            chart_df = filter_non_pitchers(season_df)
            chart_col1, chart_col2 = st.columns(2)
            chart_aw = chart_df.get_column("AW").to_list() if "AW" in chart_df.columns else []
            chart_paaw = chart_df.get_column("PAAW").to_list() if "PAAW" in chart_df.columns else []
            chart_gsvl = chart_df.get_column("GSvL_pct").to_list() if "GSvL_pct" in chart_df.columns else []
            chart_gsvr = chart_df.get_column("GSvR_pct").to_list() if "GSvR_pct" in chart_df.columns else []
            chart_name = chart_df.get_column("Name").to_list() if "Name" in chart_df.columns else []
            chart_custom = [[n] for n in chart_name]

            with chart_col1:
                st.markdown("<p style='margin-bottom: -5px;'><strong>Active Weeks vs. PA per Active Week</strong></p>", unsafe_allow_html=True)
                fig1 = go.Figure(
                    go.Scatter(
                        x=chart_aw,
                        y=chart_paaw,
                        mode="markers",
                        marker=dict(color="#27245C"),
                        customdata=chart_custom,
                        text=chart_name,
                        hovertemplate="<b>%{text}</b><br>AW: %{x}<br>PAAW: %{y}<extra></extra>",
                    )
                )
                fig1.update_layout(
                    xaxis_title="Active Weeks (AW)",
                    yaxis_title="Plate Appearances per Active Week (PAAW)",
                    margin=dict(l=10, r=10, t=10, b=10),
                )
                chart_selection1 = st.plotly_chart(fig1, on_select="rerun", selection_mode="points")

            with chart_col2:
                st.markdown("<p style='margin-bottom: -5px;'><strong>Team Games Started vs. RHP and LHP</strong></p>", unsafe_allow_html=True)
                total_gs = []
                platoon_split = []
                if {"starts_r", "starts_l", "elig_r", "elig_l", "GSvR", "GSvL"}.issubset(chart_df.columns):
                    total_col = (
                        (pl.col("starts_r") + pl.col("starts_l"))
                        / (pl.col("elig_r") + pl.col("elig_l"))
                    )
                    split_col = pl.col("GSvR") - pl.col("GSvL")
                    chart_calc = chart_df.with_columns(
                        pl.when((pl.col("elig_r") + pl.col("elig_l")) > 0)
                        .then(total_col)
                        .otherwise(None)
                        .alias("total_gs_pct"),
                        split_col.alias("platoon_split_gs"),
                    )
                    total_gs = chart_calc.get_column("total_gs_pct").to_list()
                    platoon_split = chart_calc.get_column("platoon_split_gs").to_list()

                fig2 = go.Figure(
                    go.Scatter(
                        x=platoon_split,
                        y=total_gs,
                        mode="markers",
                        marker=dict(color="#27245C"),
                        customdata=chart_custom,
                        text=chart_name,
                        hovertemplate="<b>%{text}</b><br>Platoon Split GS%: %{x}<br>Total % GS: %{y}<extra></extra>",
                    )
                )
                fig2.update_layout(
                    xaxis_title="Platoon Split GS% (R - L)",
                    yaxis_title="Total % GS",
                    margin=dict(l=10, r=10, t=10, b=10),
                    shapes=[
                        dict(
                            type="line",
                            x0=0,
                            x1=0,
                            y0=0,
                            y1=1,
                            xref="x",
                            yref="paper",
                            line=dict(color="#999999", width=1, dash="dot"),
                        )
                    ],
                )
                chart_selection2 = st.plotly_chart(fig2, on_select="rerun", selection_mode="points")

    # Handle chart click - jump to Player view
    if chart_selection1 and chart_selection1.selection.points:
        clicked_name = chart_selection1.selection.points[0]["customdata"][0]
        st.session_state.selected_player = clicked_name
        st.session_state.jump_to_player = True
        st.rerun()
    if chart_selection2 and chart_selection2.selection.points:
        clicked_name = chart_selection2.selection.points[0]["customdata"][0]
        st.session_state.selected_player = clicked_name
        st.session_state.jump_to_player = True
        st.rerun()

    league_cols = keep_existing(
        ["Year", "IDfg", "Name", "Age", "Hand", "Tm", "Primary", "Eligible",
         "Total PA", "AW", "PAAW", "SAW", "PAS", "EAW",
         "GSvR_pct", "GSvL_pct", "vR", "vL", "#R", "#L",
         "wOBA", "wTmX+", "wTmX+R", "wTmX+L"],
        season_df
    )

    # st.markdown("<p style='margin-bottom: -5px;'><strong>Regular Season</strong></p>", unsafe_allow_html=True)
    st.caption(f"{season_df.height} matching players. Click ▢ to view player profile.")

    table_df = season_df.filter(
        pl.col("Primary").is_not_null() & (pl.col("Primary").cast(pl.Utf8).str.strip_chars() != "")
    )
    if "PAAW" in table_df.columns:
        table_df = table_df.sort("PAAW", descending=True)

    show_splits = team != "All" and "IsSplit" in table_df.columns
    display_rename = {
        "Hand": "Bat",
        "Primary": "Pos",
        "Eligible": "Elig",
        "Total PA": "PA",
    }

    if show_splits:
        table_pd = table_df.to_pandas()
        split_mask = table_pd["IsSplit"].fillna(0).astype(int).reset_index(drop=True) == 1
        display_pd = table_pd[league_cols].rename(columns=display_rename).reset_index(drop=True)
        styler = display_pd.style.apply(
            lambda row: ["background-color: #f6f6f6;" if split_mask.iloc[row.name] else "" for _ in row],
            axis=1,
        )
        selection = st.dataframe(
            styler,
            height=600,
            on_select="rerun",
            selection_mode="single-row",
            column_config=column_formats,
            hide_index=True,
        )
    else:
        display_df = table_df.select(league_cols).rename(display_rename)
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
        if show_splits:
            selected_name = display_pd.iloc[selected_idx]["Name"]
        else:
            selected_name = display_df.row(selected_idx, named=True)["Name"]
        st.session_state.selected_player = selected_name
        st.session_state.jump_to_player = True
        st.rerun()


if mode == "Player":
    player_pool = df_regular
    if "Primary" in df_regular.columns:
        player_pool = df_regular.filter(
            pl.col("Primary").is_not_null()
            & (pl.col("Primary").cast(pl.Utf8).str.strip_chars() != "")
        )
    names = sorted_unique_non_null(player_pool, "Name")
    # Use selected player from League view if available, otherwise default
    if st.session_state.selected_player and st.session_state.selected_player in names:
        default_name = st.session_state.selected_player
        st.session_state.selected_player = None  # Clear after use
    else:
        default_name = "Soto, Juan"
    default_index = names.index(default_name) if default_name in names else 0
    player = st.selectbox("Player", names, index=default_index, placeholder="Type to search…", key="player")
    years = sorted_unique_non_null(df_regular.filter(pl.col("Name") == player), "Year")

    d = df_regular.filter((pl.col("Name") == player) & (pl.col("Year").is_in(years))).sort("Year")
    d_all = df.filter((pl.col("Name") == player) & (pl.col("Year").is_in(years))).sort("Year")

    # Get Team26 and Pos26 from player's data (use last non-null value)
    team26 = col_last_non_null(d, "Team26") or col_last_non_null(d_all, "Team26") or col_last_non_null(d, "Tm")
    pos26 = col_last_non_null(d, "Pos26")
    hand = col_last_non_null(d, "Hand")
    age = col_last_non_null(d, "Age")
    age_2026 = compute_age_for_year(d, 2026)

    if "AW" in d.columns:
        d = d.with_columns(
            pl.when(pl.col("Year") == 2020)
            .then(pl.col("AW").cast(pl.Float64, strict=False) * 2.7)
            .otherwise(pl.col("AW").cast(pl.Float64, strict=False))
            .alias("AW")
        )

    # --- Sparklines

    st.subheader(player)
    display_age = age_2026 if age_2026 is not None else age
    st.markdown(f"<p>Tm: {team26}\u00a0\u00a0\u00a0|\u00a0\u00a0\u00a0Pos: {pos26}\u00a0\u00a0\u00a0|\u00a0\u00a0\u00a0Age: {display_age}\u00a0\u00a0\u00a0|\u00a0\u00a0\u00a0Bats: {hand}</p>", unsafe_allow_html=True)

#    st.markdown(f"<h3>{player}</h3>", unsafe_allow_html=True)
#    st.caption(f":material/calendar_today: {age}\u00a0\u00a0\u00a0\u00a0:material/back_hand: {hand}\u00a0\u00a0\u00a0\u00a0:material/people: {team26}\u00a0\u00a0\u00a0\u00a0:material/sports_baseball: {pos26}")
#    if team26 and pos26:
#        st.markdown(f"<p style='margin-top: -1rem;'><strong>{team26} ({pos26})</strong></p>", unsafe_allow_html=True)
#    elif team26:
#        st.markdown(f"<p style='margin-top: -1rem;'>{team26}</p>", unsafe_allow_html=True)
#    elif pos26:
#        st.markdown(f"<p style='margin-top: -1rem;'>({pos26})</p>", unsafe_allow_html=True)

    key_metrics = [m for m in ["AW", "PAAW", "SAW", "PAS"] if m in d.columns]

    if not key_metrics:
        st.info("No key metrics found (AW/PAAW/SAW/PAS/EAW).")
    else:
        cols = st.columns(len(key_metrics))

        # Ensure consistent ordering by year
        d2 = d.sort("Year")
        years = d2.get_column("Year").to_list() if "Year" in d2.columns else []

        for col, m in zip(cols, key_metrics):
            with col:
                # "Card"
                with st.container(border=True):
                    latest = col_last_non_null(d2, m)
                    st.caption(m)
                    idx = tier_for(m, float(latest)) if latest is not None else None
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
                        dtype = d2.schema.get(m)
                        if is_numeric_dtype(dtype):
                            st.markdown(f"### {float(latest):.2f}")
                        else:
                            st.markdown(f"### {latest}")

                    # sparkline
                    # keep only numeric for charting
                    dtype = d2.schema.get(m)
                    if is_numeric_dtype(dtype):
                        y_range = BENCHMARKS.get(m, {}).get("display_range")
                        fig = sparkline(d2.get_column(m).to_list(), x=years, y_range=y_range)
                        st.plotly_chart(fig, config={"displayModeBar": False})

    st.toggle(
        "Show partial seasons",
        value=st.session_state.show_partial_seasons,
        key="show_partial_seasons",
    )

    d_tables = d
    if st.session_state.show_partial_seasons and "IsSplit" in d_all.columns:
        d_partial = d_all.filter(pl.col("IsSplit") == 1)
        if d_partial.height > 0:
            d_rollup = d.with_columns(pl.lit(0).alias("_row_order"))
            d_partial = d_partial.with_columns(pl.lit(1).alias("_row_order"))
            sort_cols = ["Year", "_row_order"]
            if "team_seq" in d_all.columns:
                sort_cols.append("team_seq")
            sort_cols.append("Tm")
            d_tables = pl.concat([d_rollup, d_partial], how="diagonal_relaxed").sort(sort_cols)

    # Projection row for 2026 (placeholder formula)
    # Commented out per request.
    # proj_year = 2026
    # proj_params = {
    #     "AW": {"base": 27.0, "w2025": 1.0, "w2024": 1.0, "wbase": 4.0},
    #     "PAAW": {"base": 22.0, "w2025": 4.0, "w2024": 1.0, "wbase": 0.0},
    #     "SAW": {"base": 4.9, "w2025": 4.0, "w2024": 1.0, "wbase": 0.0},
    #     "PAS": {"base": 4.2, "w2025": 4.0, "w2024": 1.0, "wbase": 1.0},
    # }
    # 
    # def value_for_year(frame: pl.DataFrame, year: int, col: str) -> float | None:
    #     if col not in frame.columns or "Year" not in frame.columns:
    #         return None
    #     sub = frame.filter(pl.col("Year") == year)
    #     if sub.height == 0:
    #         return None
    #     v = sub.get_column(col).drop_nulls()
    #     return float(v[0]) if len(v) > 0 else None
    #
    # proj_row = {"Year": proj_year, "Tm": team26, "Team26": team26, "Age": age_2026}
    # has_2025 = False
    # for stat, params in proj_params.items():
    #     base = params["base"]
    #     w2025 = params["w2025"]
    #     w2024 = params["w2024"]
    #     wbase = params["wbase"]
    #     denom = w2025 + w2024 + wbase
    #
    #     v2025 = value_for_year(d, 2025, stat)
    #     if v2025 is None:
    #         continue
    #     has_2025 = True
    #     v2024 = value_for_year(d, 2024, stat)
    #     if v2024 is None:
    #         v2024 = base
    #     if age_2026 is None:
    #         proj_val = None
    #     else:
    #         proj_val = ((w2025 * v2025 + w2024 * v2024 + wbase * base) / denom) * (1+ (27 - age_2026) * 0.00) # no age adjustment for now
    #     proj_row[stat] = proj_val
    #
    # if "AW" in proj_row and "PAAW" in proj_row and proj_row["AW"] is not None and proj_row["PAAW"] is not None:
    #     proj_row["Total PA"] = proj_row["AW"] * proj_row["PAAW"]
    #
    # if has_2025:
    #     if "_row_order" in d_tables.columns:
    #         proj_row["_row_order"] = 2
    #         sort_cols = ["Year", "_row_order"]
    #         if "team_seq" in d_tables.columns:
    #             sort_cols.append("team_seq")
    #         sort_cols.append("Tm")
    #         d_tables = pl.concat([d_tables, pl.DataFrame([proj_row])], how="diagonal_relaxed").sort(sort_cols)
    #     else:
    #         d_tables = pl.concat([d_tables, pl.DataFrame([proj_row])], how="diagonal_relaxed").sort(["Year", "Tm"])

    use_split_styles = st.session_state.show_partial_seasons and "IsSplit" in d_tables.columns
    if use_split_styles:
        d_tables_pd = d_tables.to_pandas()
        split_mask = d_tables_pd["IsSplit"].fillna(0).astype(int).reset_index(drop=True) == 1

    # --- Building Blocks
    st.markdown("**Building Blocks**")
    cols = keep_existing(["Age", "Tm", "Total PA", "AW", "PAAW", "SAW", "PAS", "EAW"], d_tables)
    if use_split_styles:
        display_df = d_tables_pd[["Year"] + cols].reset_index(drop=True)
        styler = display_df.style.apply(
            lambda row: [
                "background-color: #f6f6f6;" if split_mask.iloc[row.name] else ""
                for _ in row
            ],
            axis=1,
        )
        st.dataframe(
            styler,
            column_config=column_formats,
            hide_index=True,
            height=full_table_height(len(display_df)),
        )
    else:
        st.dataframe(
            d_tables.select(["Year"] + cols),
            column_config=column_formats,
            hide_index=True,
            height=full_table_height(d_tables.height),
        )

    # --- Handedness Splits
    st.markdown("**Handedness Splits**")
    cols = keep_existing(["Age", "Tm", "GSvR_pct", "GSvL_pct", "vR", "vL", "#R", "#L"], d_tables)
    if use_split_styles:
        display_df = d_tables_pd[["Year"] + cols].reset_index(drop=True)
        styler = display_df.style.apply(
            lambda row: [
                "background-color: #f6f6f6;" if split_mask.iloc[row.name] else ""
                for _ in row
            ],
            axis=1,
        )
        st.dataframe(
            styler,
            column_config=column_formats,
            hide_index=True,
            height=full_table_height(len(display_df)),
        )
    else:
        st.dataframe(
            d_tables.select(["Year"] + cols),
            column_config=column_formats,
            hide_index=True,
            height=full_table_height(d_tables.height),
        )

    # --- Performance
    st.markdown("**Performance**")
    cols = keep_existing(
        ["Age", "Tm", "wOBA", "wOBA vR", "wOBA vL", "wTmX+", "wTmX+R", "wTmX+L"],
        d_tables,
    )
    if use_split_styles:
        display_df = d_tables_pd[["Year"] + cols].reset_index(drop=True)
        styler = display_df.style.apply(
            lambda row: [
                "background-color: #f6f6f6;" if split_mask.iloc[row.name] else ""
                for _ in row
            ],
            axis=1,
        )
        st.dataframe(
            styler,
            column_config=column_formats,
            hide_index=True,
            height=full_table_height(len(display_df)),
        )
    else:
        st.dataframe(
            d_tables.select(["Year"] + cols),
            column_config=column_formats,
            hide_index=True,
            height=full_table_height(d_tables.height),
        )
#    with tab3:
#        cols = keep_existing(["AB/AW", "H/AW", "R/AW", "HR/AW", "RBI/AW", "SB/AW"], d)
#        st.dataframe(d[["Year"] + cols], column_config=column_formats)
