import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from difflib import SequenceMatcher
from datetime import time, datetime, date, timedelta

# ------------------------------------------------
# CONFIG
# ------------------------------------------------
APP_TITLE = "📞 TalkTime App — Fixed (Pre-loaded calling_DB.csv)"
TZ = "Asia/Kolkata"
DATA_PATH = "calling_DB.csv"   # keep this CSV beside this script or change path.

st.set_page_config(page_title="TalkTime App", layout="wide")

# ------------------------------------------------
# HELPERS
# ------------------------------------------------

def to_seconds(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        return float(x)
    s = str(x).strip()
    try:
        return float(s)
    except:
        pass
    parts = s.split(":")
    try:
        parts = [float(p) for p in parts]
    except:
        return np.nan
    if len(parts) == 2:
        m, s = parts
        return m * 60 + s
    if len(parts) == 3:
        h, m, s = parts
        return h * 3600 + m * 60 + s
    return np.nan

def parse_date_best(series):
    d1 = pd.to_datetime(series, errors="coerce", dayfirst=True)
    d2 = pd.to_datetime(series, errors="coerce", dayfirst=False)
    return d1 if d1.notna().sum() >= d2.notna().sum() else d2

def combine_date_time(date_col, time_col):
    d = parse_date_best(date_col)
    t_try = pd.to_datetime(time_col, errors="coerce")
    t = t_try.dt.time if hasattr(t_try, "dt") else None
    cat = pd.DataFrame(
        {
            "d": d.dt.date.astype(str),
            "t": pd.Series(t, index=d.index, dtype="object").astype(str),
        }
    ).agg(" ".join, axis=1)
    dt = pd.to_datetime(cat, errors="coerce")
    try:
        dt_local = dt.dt.tz_localize(TZ, nonexistent="NaT", ambiguous="NaT")
    except Exception:
        dt_local = pd.to_datetime(dt, errors="coerce").dt.tz_localize(
            TZ, nonexistent="NaT", ambiguous="NaT"
        )
    return dt_local

def norm_name(s):
    if pd.isna(s):
        return ""
    s = str(s).lower().strip()
    keep = []
    for ch in s:
        if ch.isalnum():
            keep.append(ch)
        elif ch.isspace():
            keep.append(" ")
    s = "".join(keep)
    s = " ".join(s.split())
    return s

def fuzzy_match_any(name, targets_norm, ratio_cut=0.85):
    n = norm_name(name)
    if not n:
        return False
    for t in targets_norm:
        if not t:
            continue
        if t in n or n in t:
            return True
        if any(tok and tok in n for tok in t.split()):
            return True
        if SequenceMatcher(None, n, t).ratio() >= ratio_cut:
            return True
    return False

def clean_str_col(series):
    if series is None:
        return series
    s = series.copy()
    try:
        return s.where(s.isna(), s.astype(str).str.strip())
    except Exception:
        return s

def agg_summary(df, dims, duration_field):
    """
    Group by dims and compute:
    - Total Calls
    - Total Duration (hr)     [sum in hours, 2 decimals]
    - Avg Duration (min)      [mean in minutes, 2 decimals]
    - Median Duration (min)   [median in minutes, 2 decimals]
    """
    if df.empty:
        return pd.DataFrame(
            columns=dims
            + [
                "Total Calls",
                "Total Duration (hr)",
                "Avg Duration (min)",
                "Median Duration (min)",
            ]
        )

    g = (
        df.groupby(dims, dropna=False)[duration_field]
        .agg(["count", "sum", "mean", "median"])
        .reset_index()
        .rename(columns={"count": "Total Calls"})
    )

    g["Total Duration (hr)"] = (g["sum"] / 3600).round(2)
    g["Avg Duration (min)"] = (g["mean"] / 60).round(2)
    g["Median Duration (min)"] = (g["median"] / 60).round(2)

    g = g.sort_values(
        ["Total Calls", "Total Duration (hr)"],
        ascending=[False, False],
    )

    g = g[
        dims
        + [
            "Total Calls",
            "Total Duration (hr)",
            "Avg Duration (min)",
            "Median Duration (min)",
        ]
    ]
    return g

def download_df(df, filename, label="Download CSV"):
    if df is None or df.empty:
        return
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(label, csv, file_name=filename, mime="text/csv")

# ------------------------------------------------
# TEAM DEFINITIONS (UPDATED)
# ------------------------------------------------

# B2C team
B2C_TARGETS_RAW = [
    "Aniket Srivastava",
    "Ankush Kumar",
    "Jay Nayak",
    "Ria Arora",
    "Shahbaz Ali",
    "Ziyaulhaq Badr",
    "Fuzail Saudagar",
    "kamaldeep singh",
    "Unmesh Kamble",
    "Vikas",
]

# MT team (only these)
MT_TARGETS_RAW = [
    "Niharika Mainali",
    "Ruhi Sharma",
    "VISAKHA ...",
    "TEJAS SINGH",
    "ayushman jetlearn",
    "Shujaat Shafqat",
]

B2C_TARGETS = [norm_name(x) for x in B2C_TARGETS_RAW]
MT_TARGETS = [norm_name(x) for x in MT_TARGETS_RAW]

def mask_for_targets(frame, col, targets_norm):
    if col not in frame.columns:
        return pd.Series(False, index=frame.index)
    return frame[col].apply(lambda x: fuzzy_match_any(x, targets_norm))

# ------------------------------------------------
# LOAD DATA (FIXED FILE)
# ------------------------------------------------

try:
    df = pd.read_csv(DATA_PATH, low_memory=False)
except Exception as e:
    st.error(f"❌ Could not load `{DATA_PATH}`. Please ensure it's present.\n\nError: {e}")
    st.stop()

# Basic cleaning
for col in ["Caller", "Country Name", "Call Type", "Call Status"]:
    if col in df.columns:
        df[col] = clean_str_col(df[col])

df["_duration_sec"] = (
    df["Call Duration"].apply(to_seconds)
    if "Call Duration" in df.columns
    else np.nan
)

df["_date_parsed"] = parse_date_best(df["Date"]) if "Date" in df.columns else pd.NaT
df["_date_only"] = df["_date_parsed"].dt.date

if {"Date", "Time"}.issubset(df.columns):
    df["_dt_local"] = combine_date_time(df["Date"], df["Time"])
    df["_hour"] = df["_dt_local"].dt.hour
else:
    df["_dt_local"] = pd.NaT
    df["_hour"] = np.nan

# Data window based on file
if df["_date_only"].notna().any():
    data_min = df["_date_only"].min()
    data_max = df["_date_only"].max()
else:
    data_min = date.today()
    data_max = date.today()

# ------------------------------------------------
# SIDEBAR
# ------------------------------------------------

st.title(APP_TITLE)

with st.sidebar:
    st.header("Data Source")
    st.caption(f"Using pre-loaded **{DATA_PATH}** (no manual upload).")
    st.caption(f"Data window: **{data_min} → {data_max}**")

    st.header("1) Agent Set")
    team_mode = st.radio(
        "Analyze:",
        ["All agents", "B2C team only", "MT Team only"],
    )

    st.header("2) Calls Mode")
    mode = st.radio(
        "Calls to include",
        ["All calls", "Only calls with duration ≥ threshold"],
        index=0,
    )
    threshold = st.slider(
        "Threshold (sec)",
        10,
        300,
        60,
        5,
    )

    st.header("3) Period (IST)")
    preset = st.radio(
        "Pick a range",
        ["Today (max file date)", "Yesterday (vs file)", "Custom"],
        index=0,
    )

    if preset == "Custom":
        default_start = data_min
        default_end = data_max
        custom_dates = st.date_input(
            "Custom dates (Start & End, inclusive)",
            value=(default_start, default_end),
        )
    else:
        custom_dates = None

    st.caption("Time range within selected dates:")
    t_start = st.time_input("Start time", value=time(0, 0, 0), key="t_start")
    t_end = st.time_input("End time", value=time(23, 59, 59), key="t_end")

    include_missing_time = st.checkbox(
        "Include rows with missing Time (use Date only)",
        value=True,
    )

    st.header("4) Filters")
    include_missing = st.checkbox(
        "Include blank values for filtered columns",
        value=True,
    )

    if "Caller" in df.columns:
        agents = sorted(df["Caller"].dropna().astype(str).unique().tolist())
        sel_agents = st.multiselect("Agent(s)", agents, default=agents)
    else:
        sel_agents = None

    if "Country Name" in df.columns:
        countries = sorted(df["Country Name"].dropna().astype(str).unique().tolist())
        sel_countries = st.multiselect("Country(ies)", countries, default=countries)
    else:
        sel_countries = None

    if "Call Type" in df.columns:
        call_types = sorted(df["Call Type"].dropna().astype(str).unique().tolist())
        sel_types = st.multiselect("Call Type(s)", call_types, default=call_types)
    else:
        sel_types = None

    if "Call Status" in df.columns:
        statuses = sorted(df["Call Status"].dropna().astype(str).unique().tolist())
        sel_status = st.multiselect("Call Status", statuses, default=statuses)
    else:
        sel_status = None

# ------------------------------------------------
# DATE + TIME FILTER (USING FILE DATES)
# ------------------------------------------------

if preset == "Today (max file date)":
    start_date = data_max
    end_date = data_max
elif preset == "Yesterday (vs file)":
    start_date = data_max - timedelta(days=1)
    end_date = data_max - timedelta(days=1)
else:
    if isinstance(custom_dates, (list, tuple)) and len(custom_dates) == 2:
        start_date, end_date = custom_dates
    else:
        start_date, end_date = data_min, data_max

# clamp
if start_date < data_min:
    start_date = data_min
if end_date > data_max:
    end_date = data_max
if end_date < start_date:
    end_date = start_date

start_dt = pd.Timestamp(
    datetime.combine(start_date, t_start)
).tz_localize(TZ)
end_dt = pd.Timestamp(
    datetime.combine(end_date, t_end)
).tz_localize(TZ)

df_f = df.copy()

if "_dt_local" in df_f.columns:
    dt_mask = (
        df_f["_dt_local"].notna()
        & (df_f["_dt_local"] >= start_dt)
        & (df_f["_dt_local"] <= end_dt)
    )
else:
    dt_mask = pd.Series(False, index=df_f.index)

date_mask = (
    df_f["_date_only"].notna()
    & (df_f["_date_only"] >= start_date)
    & (df_f["_date_only"] <= end_date)
)

if include_missing_time:
    final_time_mask = dt_mask | (df_f["_dt_local"].isna() & date_mask)
else:
    final_time_mask = dt_mask

df_f = df_f[final_time_mask].copy()

# ------------------------------------------------
# TEAM FILTER
# ------------------------------------------------

if team_mode == "B2C team only":
    m = mask_for_targets(df_f, "Caller", B2C_TARGETS)
    if include_missing:
        m = m | df_f["Caller"].isna()
    df_f = df_f[m].copy()
elif team_mode == "MT Team only":
    m = mask_for_targets(df_f, "Caller", MT_TARGETS)
    if include_missing:
        m = m | df_f["Caller"].isna()
    df_f = df_f[m].copy()

# ------------------------------------------------
# COLUMN FILTERS
# ------------------------------------------------

def apply_filter(frame, col, selections):
    if selections is None or col not in frame.columns:
        return frame
    if len(selections) == 0:
        return frame
    col_vals = frame[col].astype(str)
    if include_missing:
        return frame[col_vals.isin(selections) | frame[col].isna()]
    else:
        return frame[col_vals.isin(selections)]

df_f = apply_filter(df_f, "Caller", sel_agents)
df_f = apply_filter(df_f, "Country Name", sel_countries)
df_f = apply_filter(df_f, "Call Type", sel_types)
df_f = apply_filter(df_f, "Call Status", sel_status)

# Calls mode
if mode.startswith("Only"):
    df_view = df_f[df_f["_duration_sec"] >= float(threshold)].copy()
else:
    df_view = df_f.copy()

# ------------------------------------------------
# OVERVIEW KPIs
# ------------------------------------------------

st.subheader("Overview")

if df_view.empty:
    st.warning(
        "No records in the selected filters/date-time window. "
        "Try expanding the date range or switching presets."
    )

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Calls", f"{len(df_view):,}")

if df_view["_duration_sec"].notna().any():
    avg_min = (df_view["_duration_sec"].mean() / 60).round(2)
    med_min = (df_view["_duration_sec"].median() / 60).round(2)
    k2.metric("Avg Duration (min)", f"{avg_min:,.2f}")
    k3.metric("Median Duration (min)", f"{med_min:,.2f}")
else:
    k2.metric("Avg Duration (min)", "NA")
    k3.metric("Median Duration (min)", "NA")

k4.metric(
    "Agents",
    df_view["Caller"].nunique() if "Caller" in df_view.columns else 0
)

st.caption(
    f"Team: **{team_mode}** | Calls: **{mode}** | "
    f"Threshold: **{threshold}s** | "
    f"Window: **{start_dt.strftime('%Y-%m-%d %H:%M')} → {end_dt.strftime('%Y-%m-%d %H:%M')} IST**"
)

# ------------------------------------------------
# TABS
# ------------------------------------------------

tab1, tab2, tab3, tab4 = st.tabs(
    ["Agent-wise", "Country-wise", "Agent × Country", "24h Engagement"]
)

with tab1:
    st.markdown("### Agent-wise Summary")
    if "Caller" in df_view.columns:
        agg = agg_summary(df_view, ["Caller"], "_duration_sec")
        st.dataframe(agg, use_container_width=True)
        download_df(agg, "agent_wise_calls.csv")
        if not agg.empty:
            chart = (
                alt.Chart(agg)
                .mark_bar()
                .encode(
                    x=alt.X("Caller:N", sort="-y", title="Agent"),
                    y=alt.Y("Total Calls:Q"),
                    tooltip=[
                        "Caller",
                        "Total Calls",
                        "Total Duration (hr)",
                        "Avg Duration (min)",
                        "Median Duration (min)",
                    ],
                )
                .properties(height=360)
                .interactive()
            )
            st.altair_chart(chart, use_container_width=True)
    else:
        st.info("Missing 'Caller' column.")

with tab2:
    st.markdown("### Country-wise Summary")
    if "Country Name" in df_view.columns:
        agg = agg_summary(df_view, ["Country Name"], "_duration_sec")
        st.dataframe(agg, use_container_width=True)
        download_df(agg, "country_wise_calls.csv")
        if not agg.empty:
            chart = (
                alt.Chart(agg)
                .mark_bar()
                .encode(
                    x=alt.X("Country Name:N", sort="-y", title="Country"),
                    y=alt.Y("Total Calls:Q"),
                    tooltip=[
                        "Country Name",
                        "Total Calls",
                        "Total Duration (hr)",
                        "Avg Duration (min)",
                        "Median Duration (min)",
                    ],
                )
                .properties(height=360)
                .interactive()
            )
            st.altair_chart(chart, use_container_width=True)
    else:
        st.info("Missing 'Country Name' column.")

with tab3:
    st.markdown("### Agent × Country Matrix")
    if {"Caller", "Country Name"}.issubset(df_view.columns):
        agg = agg_summary(df_view, ["Caller", "Country Name"], "_duration_sec")
        st.dataframe(agg, use_container_width=True)
        download_df(agg, "agent_country_matrix.csv")
        if not agg.empty:
            chart = (
                alt.Chart(agg)
                .mark_bar()
                .encode(
                    x=alt.X("Caller:N", sort="-y", title="Agent"),
                    y=alt.Y("Total Calls:Q"),
                    color=alt.Color("Country Name:N", title="Country"),
                    tooltip=[
                        "Caller",
                        "Country Name",
                        "Total Calls",
                        "Total Duration (hr)",
                    ],
                )
                .properties(height=380)
                .interactive()
            )
            st.altair_chart(chart, use_container_width=True)
    else:
        st.info("Need both 'Caller' and 'Country Name' for this view.")

with tab4:
    st.markdown("### 24h Engagement")

    df_time = (
        df_view.dropna(subset=["_hour"])
        if "_hour" in df_view.columns
        else pd.DataFrame()
    )

    if not df_time.empty:
        # Attempts by hour (always full)
        attempts = (
            df_time.groupby("_hour")
            .size()
            .reset_index(name="Attempts")
            .rename(columns={"_hour": "Hour"})
        )

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Attempts by Hour (All)**")
            st.dataframe(
                attempts.sort_values("Hour"),
                use_container_width=True,
            )
            download_df(
                attempts.sort_values("Hour"),
                "attempts_by_hour.csv",
            )

        with c2:
            chart = (
                alt.Chart(attempts)
                .mark_circle()
                .encode(
                    x=alt.X("Hour:O", title="Hour (0–23, IST)"),
                    y=alt.Y("Attempts:Q"),
                    size=alt.Size("Attempts:Q", legend=None),
                    tooltip=["Hour:O", "Attempts:Q"],
                )
                .properties(height=340)
                .interactive()
            )
            st.altair_chart(chart, use_container_width=True)

        st.divider()

        # N selector for Bubble + Heatmap
        max_attempts = int(df_time.shape[0]) if df_time.shape[0] > 0 else 1
        n_threshold = st.number_input(
            "Show only cells with Attempts ≥ N (for Bubble & Heatmap)",
            min_value=1,
            max_value=max_attempts,
            value=1,
            step=1,
        )

        # Bubble: Hour × Country (filtered)
        if "Country Name" in df_time.columns:
            st.markdown("**Hour × Country (Bubble)**")
            a2 = (
                df_time.groupby(["_hour", "Country Name"])
                .size()
                .reset_index(name="Attempts")
                .rename(columns={"_hour": "Hour"})
            )
            a2 = a2[a2["Attempts"] >= n_threshold]

            if not a2.empty:
                bubble = (
                    alt.Chart(a2)
                    .mark_circle()
                    .encode(
                        x=alt.X("Hour:O"),
                        y=alt.Y("Country Name:N", title="Country"),
                        size=alt.Size("Attempts:Q", legend=None),
                        tooltip=[
                            "Hour:O",
                            "Country Name:N",
                            "Attempts:Q",
                        ],
                    )
                    .properties(height=420)
                    .interactive()
                )
                st.altair_chart(bubble, use_container_width=True)
                download_df(
                    a2.sort_values(["Country Name", "Hour"]),
                    "hour_country_bubble_filtered.csv",
                )
            else:
                st.info("No Hour × Country cells meet the selected N threshold.")

        st.divider()

        # Heatmap: Agent × Hour (filtered)
        if "Caller" in df_time.columns:
            st.markdown("**Agent × Hour (Heatmap)**")
            hh = (
                df_time.groupby(["Caller", "_hour"])
                .size()
                .reset_index(name="Attempts")
                .rename(columns={"_hour": "Hour"})
            )
            hh = hh[hh["Attempts"] >= n_threshold]

            if not hh.empty:
                heat = (
                    alt.Chart(hh)
                    .mark_rect()
                    .encode(
                        x=alt.X("Hour:O"),
                        y=alt.Y("Caller:N", title="Agent"),
                        color=alt.Color("Attempts:Q"),
                        tooltip=["Caller", "Hour", "Attempts"],
                    )
                    .properties(height=420)
                    .interactive()
                )
                st.altair_chart(heat, use_container_width=True)
                download_df(
                    hh.sort_values(["Caller", "Hour"]),
                    "agent_hour_heatmap_filtered.csv",
                )
            else:
                st.info("No Agent × Hour cells meet the selected N threshold.")
    else:
        st.info("No valid time data in current filter to show 24h engagement.")

st.caption(
    "Teams mapped as per latest list. "
    "Durations: Total Duration (hr), Avg/Median Duration (min). "
    "24h Engagement Bubble & Heatmap respect Attempts ≥ N."
)
