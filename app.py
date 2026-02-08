import io
import warnings

import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(page_title="Local Business Analytics Dashboard", page_icon="📊", layout="wide")

st.title("Local Business Analytics Dashboard")
st.caption(
    "Upload a CSV to explore summary statistics, filter data, visualize trends, and export results."
)


@st.cache_data(show_spinner=False)
def read_csv_bytes(file_bytes: bytes) -> tuple[pd.DataFrame, dict]:
    """Read CSV from bytes with encoding fallbacks and delimiter sniffing."""
    meta: dict[str, str] = {}
    encodings = ["utf-8", "utf-8-sig", "latin-1"]

    for enc in encodings:
        try:
            buf = io.BytesIO(file_bytes)
            df = pd.read_csv(buf, sep=None, engine="python", encoding=enc)
            meta["encoding"] = enc
            meta["delimiter"] = "auto"
            return df, meta
        except Exception:
            continue

    buf = io.BytesIO(file_bytes)
    df = pd.read_csv(buf)
    meta["encoding"] = "default"
    meta["delimiter"] = "default"
    return df, meta


def coerce_datetime_columns(df: pd.DataFrame, threshold: float = 0.85) -> pd.DataFrame:
    """Convert object columns that parse as datetimes above a threshold."""
    out = df.copy()
    for col in out.columns:
        if out[col].dtype != "object":
            continue
        sample = out[col].dropna().astype(str).head(500)
        if sample.empty:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parsed = pd.to_datetime(sample, errors="coerce")
        if float(parsed.notna().mean()) >= threshold:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                out[col] = pd.to_datetime(out[col], errors="coerce")
    return out


def column_types(df: pd.DataFrame) -> tuple[list[str], list[str], list[str], list[str]]:
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    datetime_cols = df.select_dtypes(include=["datetime64[ns]", "datetimetz"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    other_cols = [
        col
        for col in df.columns
        if col not in set(numeric_cols + datetime_cols + categorical_cols)
    ]
    return numeric_cols, categorical_cols, datetime_cols, other_cols


def dataset_overview_table(df: pd.DataFrame) -> pd.DataFrame:
    missing = df.isna().sum()
    missing_pct = (df.isna().mean() * 100).round(2)
    unique_values = df.nunique(dropna=True)
    dtype = df.dtypes.astype(str)
    overview = pd.DataFrame(
        {
            "dtype": dtype,
            "non_null": df.shape[0] - missing,
            "missing_count": missing,
            "missing_pct": missing_pct,
            "unique_values": unique_values,
        }
    )
    return (
        overview.reset_index()
        .rename(columns={"index": "column"})
        .sort_values(["missing_pct", "unique_values"], ascending=[False, False])
    )


def numeric_stats(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    if not numeric_cols:
        return pd.DataFrame(columns=["column", "mean", "median", "std_dev", "min", "max"])
    data = df[numeric_cols]
    stats = pd.DataFrame(
        {
            "mean": data.mean(numeric_only=True),
            "median": data.median(numeric_only=True),
            "std_dev": data.std(numeric_only=True),
            "min": data.min(numeric_only=True),
            "max": data.max(numeric_only=True),
        }
    )
    return stats.reset_index().rename(columns={"index": "column"})


def generate_findings(
    df_full: pd.DataFrame,
    df_filtered: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
    datetime_cols: list[str],
) -> list[str]:
    insights: list[str] = []
    total_rows = len(df_full)
    filtered_rows = len(df_filtered)

    if total_rows and filtered_rows != total_rows:
        pct = (filtered_rows / total_rows) * 100
        insights.append(
            f"Filtering impact: viewing **{filtered_rows:,}** of **{total_rows:,}** rows ({pct:.1f}%)."
        )

    missing_pct = (df_full.isna().mean() * 100).sort_values(ascending=False)
    if not missing_pct.empty and float(missing_pct.iloc[0]) > 0:
        col = missing_pct.index[0]
        insights.append(
            f"Missing data: **{col}** has **{float(missing_pct.iloc[0]):.1f}%** missing values."
        )

    if total_rows > 0:
        dup_pct = float(df_full.duplicated().mean() * 100)
        if dup_pct > 0:
            insights.append(f"Duplicates: **{dup_pct:.1f}%** of rows are duplicates.")

    if datetime_cols and not df_filtered.empty:
        best_col = None
        best_span = None
        for col in datetime_cols:
            series = df_filtered[col].dropna()
            if series.empty:
                continue
            span = series.max() - series.min()
            if best_span is None or span > best_span:
                best_col = col
                best_span = span
        if best_col:
            series = df_filtered[best_col].dropna()
            insights.append(
                "Date range: **{col}** spans from **{start}** to **{end}** (filtered view).".format(
                    col=best_col,
                    start=series.min().date(),
                    end=series.max().date(),
                )
            )

    if categorical_cols and not df_filtered.empty:
        best_col = None
        best_share = None
        best_value = None
        for col in categorical_cols:
            series = df_filtered[col].fillna("(Missing)").astype(str)
            counts = series.value_counts()
            if counts.empty:
                continue
            share = float(counts.iloc[0] / counts.sum())
            if best_share is None or share > best_share:
                best_col = col
                best_share = share
                best_value = str(counts.index[0])
        if best_col and best_share is not None and best_value is not None:
            insights.append(
                f"Top category: In **{best_col}**, **{best_value}** appears **{best_share * 100:.1f}%** of the time."
            )

    if numeric_cols and not df_filtered.empty:
        best_num = None
        best_outlier_pct = None
        for col in numeric_cols:
            values = df_filtered[col].dropna()
            if values.shape[0] < 8:
                continue
            q1 = float(values.quantile(0.25))
            q3 = float(values.quantile(0.75))
            iqr = q3 - q1
            if iqr == 0:
                continue
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outlier_pct = float(((values < lower) | (values > upper)).mean() * 100)
            if best_outlier_pct is None or outlier_pct > best_outlier_pct:
                best_num = col
                best_outlier_pct = outlier_pct
        if best_num and best_outlier_pct is not None and best_outlier_pct > 0:
            insights.append(
                f"Outliers: **{best_outlier_pct:.1f}%** of values in **{best_num}** look like outliers (IQR rule)."
            )
        else:
            for col in numeric_cols:
                values = df_filtered[col].dropna()
                if not values.empty:
                    insights.append(
                        f"Value summary: **{col}** mean is **{values.mean():.3g}** with median **{values.median():.3g}**."
                    )
                    break

    if len(insights) < 3:
        if df_filtered.empty:
            insights.append("No findings available because the filtered dataset is empty.")
        else:
            insights.append(f"Columns available: **{df_filtered.shape[1]}**.")
            insights.append("Tip: Use the sidebar filters to slice the dataset for sharper insights.")

    return insights[:3]


def apply_filters_sidebar(
    df: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
    datetime_cols: list[str],
) -> pd.DataFrame:
    st.sidebar.header("Filters")
    st.sidebar.caption("Pick columns to filter. Each filter tightens the dataset.")

    filterable_cols = categorical_cols + numeric_cols + datetime_cols
    selected_cols = st.sidebar.multiselect(
        "Filter columns",
        options=filterable_cols,
        default=[],
        help="Choose 1-3 filters for a clean view.",
    )

    filtered = df.copy()

    with st.sidebar.expander("Filter settings", expanded=True):
        for col in selected_cols:
            key_base = f"f_{abs(hash(col))}"

            if col in categorical_cols:
                st.markdown(f"**{col}**")
                series = filtered[col].fillna("(Missing)").astype(str)
                top = series.value_counts().head(50)
                options = top.index.tolist()

                chosen = st.multiselect(
                    "Pick values (top 50 shown)",
                    options=options,
                    default=[],
                    key=f"{key_base}_cat_vals",
                )
                contains = st.text_input(
                    "Contains text (optional)",
                    value="",
                    key=f"{key_base}_cat_contains",
                )

                if chosen:
                    filtered = filtered[filtered[col].fillna("(Missing)").astype(str).isin(chosen)]
                if contains.strip():
                    filtered = filtered[
                        filtered[col].astype(str).str.contains(contains.strip(), case=False, na=False)
                    ]

                st.caption(f"Rows after {col}: {len(filtered):,}")

            elif col in numeric_cols:
                st.markdown(f"**{col}**")
                series = filtered[col].dropna()
                if series.empty:
                    st.info("No numeric values available to filter here.")
                    continue

                min_val = float(series.min())
                max_val = float(series.max())

                if min_val == max_val:
                    st.write(f"All values are {min_val}. No range filter needed.")
                    continue

                rng = st.slider(
                    "Range",
                    min_value=min_val,
                    max_value=max_val,
                    value=(min_val, max_val),
                    key=f"{key_base}_num_rng",
                )
                keep_missing = st.checkbox(
                    "Keep missing values",
                    value=True,
                    key=f"{key_base}_num_keepna",
                )

                mask = filtered[col].between(rng[0], rng[1], inclusive="both")
                if keep_missing:
                    mask = mask | filtered[col].isna()
                filtered = filtered[mask]

                st.caption(f"Rows after {col}: {len(filtered):,}")

            elif col in datetime_cols:
                st.markdown(f"**{col}**")
                series = filtered[col].dropna()
                if series.empty:
                    st.info("No datetime values available to filter here.")
                    continue

                min_dt = series.min()
                max_dt = series.max()
                if pd.isna(min_dt) or pd.isna(max_dt) or min_dt == max_dt:
                    st.write("Not enough date variation to filter.")
                    continue

                picked = st.date_input(
                    "Date range",
                    value=(min_dt.date(), max_dt.date()),
                    min_value=min_dt.date(),
                    max_value=max_dt.date(),
                    key=f"{key_base}_dt_rng",
                )

                if isinstance(picked, (list, tuple)) and len(picked) == 2:
                    start_date, end_date = picked
                    start_ts = pd.to_datetime(start_date)
                    end_ts = pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

                    keep_missing = st.checkbox(
                        "Keep missing dates",
                        value=True,
                        key=f"{key_base}_dt_keepna",
                    )

                    mask = (filtered[col] >= start_ts) & (filtered[col] <= end_ts)
                    if keep_missing:
                        mask = mask | filtered[col].isna()
                    filtered = filtered[mask]

                st.caption(f"Rows after {col}: {len(filtered):,}")

    return filtered


uploaded = st.file_uploader("Upload a CSV file", type=["csv"])
if uploaded is None:
    st.info("Upload a CSV to begin.")
    st.stop()

try:
    raw_bytes = uploaded.getvalue()
    data, meta = read_csv_bytes(raw_bytes)
except Exception as exc:
    st.error("Could not read this file as a CSV. Check delimiters and encoding.")
    st.exception(exc)
    st.stop()

if data is None or data.shape[1] == 0:
    st.error("This CSV has no columns. Upload a valid dataset.")
    st.stop()

data = data.dropna(how="all").copy()
if data.empty:
    st.error("Your dataset has no rows (or all rows were empty). Upload a non-empty CSV.")
    st.stop()

try:
    data = coerce_datetime_columns(data)
except Exception:
    pass

numeric_cols, categorical_cols, datetime_cols, other_cols = column_types(data)
filtered_data = apply_filters_sidebar(data, numeric_cols, categorical_cols, datetime_cols)

dup_count = int(data.duplicated().sum())
mem_mb = float(data.memory_usage(deep=True).sum() / (1024 * 1024))

m1, m2, m3, m4 = st.columns(4)
m1.metric("Total rows", f"{len(data):,}")
m2.metric("Filtered rows", f"{len(filtered_data):,}")
m3.metric("Columns", f"{data.shape[1]}")
m4.metric("Duplicate rows", f"{dup_count:,}")

st.caption(
    "Loaded using encoding: {enc} (delimiter: {delim}). Memory: {mem:.2f} MB.".format(
        enc=meta.get("encoding", "unknown"),
        delim=meta.get("delimiter", "unknown"),
        mem=mem_mb,
    )
)

st.divider()

tab_overview, tab_explore, tab_viz, tab_findings, tab_export = st.tabs(
    ["Overview", "Explore", "Visualize", "Findings", "Export"]
)

with tab_overview:
    left, right = st.columns([1.15, 0.85])

    with left:
        st.subheader("Column overview")
        overview = dataset_overview_table(data)
        st.dataframe(overview, use_container_width=True, height=420)

    with right:
        st.subheader("Numeric descriptive stats")
        if numeric_cols:
            stats = numeric_stats(data, numeric_cols)
            st.dataframe(stats, use_container_width=True, height=420)
        else:
            st.info("No numeric columns detected, so stats are not available.")

    if other_cols:
        st.caption(
            f"Note: {len(other_cols)} column(s) have uncommon types and are not used for charts by default."
        )

with tab_explore:
    st.subheader("Filtered data preview")
    if filtered_data.empty:
        st.warning("Filtered dataset is empty. Loosen filters in the sidebar.")
    else:
        show_cols = st.multiselect(
            "Columns to display",
            options=filtered_data.columns.tolist(),
            default=filtered_data.columns.tolist()[: min(8, filtered_data.shape[1])],
        )
        st.dataframe(filtered_data[show_cols].head(500), use_container_width=True, height=520)

with tab_viz:
    if filtered_data.empty:
        st.warning("No charts can be generated because the filtered dataset is empty.")
    else:
        v1, v2 = st.columns(2)

        with v1:
            st.subheader("Histogram (numeric)")
            if numeric_cols:
                hist_col = st.selectbox("Numeric column", options=numeric_cols, index=0)
                bins = st.slider("Bins", min_value=5, max_value=80, value=30)

                color_by = None
                if categorical_cols:
                    color_choices = ["(None)"] + categorical_cols
                    pick = st.selectbox("Color by (optional)", options=color_choices, index=0)
                    if pick != "(None)":
                        color_by = pick

                fig = px.histogram(
                    filtered_data,
                    x=hist_col,
                    nbins=bins,
                    color=color_by,
                    title=f"Distribution of {hist_col}",
                )
                fig.update_layout(margin=dict(l=20, r=20, t=50, b=20))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No numeric columns available for a histogram.")

        with v2:
            st.subheader("Bar chart (categorical)")
            if categorical_cols:
                cat_col = st.selectbox("Categorical column", options=categorical_cols, index=0)
                top_n = st.slider("Top categories to show", min_value=5, max_value=30, value=10)
                as_percent = st.toggle("Show as percent", value=False)

                series = filtered_data[cat_col].fillna("(Missing)").astype(str)
                vc = series.value_counts().head(top_n)
                bar_df = vc.reset_index()
                bar_df.columns = [cat_col, "count"]

                if as_percent:
                    bar_df["percent"] = bar_df["count"] / bar_df["count"].sum() * 100
                    fig = px.bar(
                        bar_df,
                        x=cat_col,
                        y="percent",
                        title=f"Top {top_n} categories for {cat_col} (%)",
                    )
                    fig.update_yaxes(ticksuffix="%")
                else:
                    fig = px.bar(
                        bar_df,
                        x=cat_col,
                        y="count",
                        title=f"Top {top_n} categories for {cat_col}",
                    )

                fig.update_layout(margin=dict(l=20, r=20, t=50, b=20))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No categorical columns available for a bar chart.")

        st.divider()

        st.subheader("Correlation (numeric)")
        if len(numeric_cols) >= 2:
            corr = filtered_data[numeric_cols].corr(numeric_only=True)
            fig = px.imshow(corr, text_auto=".2f", title="Correlation heatmap (filtered view)")
            fig.update_layout(margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Need at least 2 numeric columns for a correlation heatmap.")

        st.divider()

        st.subheader("Time series (if you have dates)")
        if datetime_cols and numeric_cols:
            c1, c2, c3 = st.columns([0.38, 0.38, 0.24])
            with c1:
                dt_col = st.selectbox("Date column", options=datetime_cols, index=0)
            with c2:
                val_col = st.selectbox("Value column", options=numeric_cols, index=0)
            with c3:
                freq = st.selectbox("Bucket", options=["D", "W", "M"], index=1)

            agg = st.selectbox("Aggregation", options=["sum", "mean", "median", "count"], index=0)

            ts = filtered_data[[dt_col, val_col]].dropna(subset=[dt_col]).copy()
            if ts.empty:
                st.info("No usable datetime rows after filtering.")
            else:
                ts = ts.set_index(dt_col).sort_index()
                if agg == "count":
                    out = ts[val_col].resample(freq).count().rename("value").reset_index()
                elif agg == "mean":
                    out = ts[val_col].resample(freq).mean().rename("value").reset_index()
                elif agg == "median":
                    out = ts[val_col].resample(freq).median().rename("value").reset_index()
                else:
                    out = ts[val_col].resample(freq).sum().rename("value").reset_index()

                fig = px.line(out, x=dt_col, y="value", title=f"{agg.title()} of {val_col} over time ({freq})")
                fig.update_layout(margin=dict(l=20, r=20, t=50, b=20))
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Time series needs at least one datetime column and one numeric column.")

with tab_findings:
    st.subheader("Top 3 insights")
    insights = generate_findings(data, filtered_data, numeric_cols, categorical_cols, datetime_cols)
    for idx, insight in enumerate(insights, start=1):
        st.write(f"{idx}. {insight}")
    st.caption("Auto-generated heuristics to surface anomalies quickly.")

with tab_export:
    st.subheader("Export filtered dataset")
    if filtered_data.empty:
        st.warning("Filtered dataset is empty, so there is nothing to export.")
    else:
        csv_bytes = filtered_data.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download filtered CSV",
            data=csv_bytes,
            file_name="filtered_data.csv",
            mime="text/csv",
        )
        st.caption(
            f"Export contains {len(filtered_data):,} rows and {filtered_data.shape[1]} columns."
        )
