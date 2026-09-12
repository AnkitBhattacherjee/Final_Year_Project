"""
DataVisualizer — Comprehensive Plotly Chart Library
====================================================
Supports 20+ chart types, all rendered locally with zero API calls.
"""

import base64
import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from typing import Any, Dict, List, Optional


_COLORS = [
    "#2563EB", "#7C3AED", "#059669", "#D97706", "#DC2626",
    "#0891B2", "#4F46E5", "#BE185D", "#0F766E", "#B45309",
    "#7E22CE", "#0369A1", "#15803D", "#C2410C",
]

_FONT = dict(family="Inter, Roboto, sans-serif", size=12, color="#334155")

_LAYOUT_BASE = dict(
    template="plotly_white",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=50, r=50, t=60, b=50),
    autosize=True,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    hoverlabel=dict(bgcolor="white", bordercolor="#E2E8F0", font_size=12),
)


def _apply_layout(fig, title=None):
    fig.update_layout(**_LAYOUT_BASE)
    fig.update_layout(font=_FONT)
    if title:
        fig.update_layout(title_text=title, title_x=0)
    return fig


def _decode_bdata(d):
    """Recursively decode Plotly bdata binary buffers and numpy types into standard Python lists/scalars."""
    if isinstance(d, dict):
        if "bdata" in d and "dtype" in d:
            try:
                raw = base64.b64decode(d["bdata"])
                arr = np.frombuffer(raw, dtype=d["dtype"])
                return arr.tolist()
            except Exception:
                pass
        return {k: _decode_bdata(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [_decode_bdata(x) for x in d]
    elif isinstance(d, np.ndarray):
        return d.tolist()
    elif isinstance(d, (np.integer, np.int64, np.int32)):
        return int(d)
    elif isinstance(d, (np.floating, np.float64, np.float32)):
        return float(d)
    return d


def _to_json(fig):
    raw = json.loads(pio.to_json(fig))
    return _decode_bdata(raw)


def _safe(val):
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    return val


class DataVisualizer:

    THEME_COLORS = _COLORS

    @classmethod
    def _apply_layout_styling(cls, fig):
        return _apply_layout(fig)

    @classmethod
    def create_chart(cls, df, chart_type, x_col, y_col=None, title=None, **kwargs):
        if df is None or (isinstance(df, pd.DataFrame) and df.empty):
            return {"error": "No data available to chart."}

        color_col = kwargs.get("color_col")
        size_col  = kwargs.get("size_col")

        ct = (chart_type or "bar").lower().replace(" ", "").replace("_", "").replace("-", "")

        if x_col and isinstance(df, pd.DataFrame) and x_col not in df.columns:
            if len(df.columns):
                x_col = df.columns[0]

        try:
            if ct in ("bar", "barchart", "verticalbar"):
                fig = px.bar(df, x=x_col, y=y_col, title=title, color_discrete_sequence=_COLORS)
                _mark_bar_values(fig)

            elif ct in ("hbar", "horizontalbar", "barh"):
                fig = px.bar(df, x=y_col, y=x_col, orientation="h", title=title, color_discrete_sequence=_COLORS)

            elif ct in ("groupedbar", "grouped", "grouped_bar"):
                if color_col and color_col in df.columns:
                    fig = px.bar(df, x=x_col, y=y_col, color=color_col, barmode="group", title=title, color_discrete_sequence=_COLORS)
                else:
                    fig = px.bar(df, x=x_col, y=y_col, title=title, color_discrete_sequence=_COLORS)

            elif ct in ("stackedbar", "stacked", "stacked_bar"):
                if color_col and color_col in df.columns:
                    fig = px.bar(df, x=x_col, y=y_col, color=color_col, barmode="stack", title=title, color_discrete_sequence=_COLORS)
                else:
                    fig = px.bar(df, x=x_col, y=y_col, barmode="stack", title=title, color_discrete_sequence=_COLORS)

            elif ct in ("line", "linechart", "rollingline", "trendline", "multiline"):
                if color_col and color_col in df.columns:
                    fig = px.line(df, x=x_col, y=y_col, color=color_col, title=title, markers=True, color_discrete_sequence=_COLORS)
                else:
                    fig = px.line(df, x=x_col, y=y_col, title=title, markers=True, color_discrete_sequence=_COLORS)

            elif ct in ("area", "areachart"):
                if color_col and color_col in df.columns:
                    fig = px.area(df, x=x_col, y=y_col, color=color_col, title=title, color_discrete_sequence=_COLORS)
                else:
                    fig = px.area(df, x=x_col, y=y_col, title=title, color_discrete_sequence=_COLORS)

            elif ct in ("stackedarea", "stacked_area"):
                if color_col and color_col in df.columns:
                    fig = px.area(df, x=x_col, y=y_col, color=color_col, title=title, color_discrete_sequence=_COLORS)
                else:
                    fig = px.area(df, x=x_col, y=y_col, title=title, color_discrete_sequence=_COLORS)

            elif ct in ("pie", "piechart"):
                fig = px.pie(df, names=x_col, values=y_col, title=title, hole=0.1, color_discrete_sequence=_COLORS)
                fig.update_traces(textposition="inside", textinfo="percent+label")

            elif ct in ("donut", "doughnut"):
                fig = px.pie(df, names=x_col, values=y_col, title=title, hole=0.5, color_discrete_sequence=_COLORS)
                fig.update_traces(textposition="inside", textinfo="percent+label")

            elif ct in ("scatter", "scatterplot"):
                if color_col and color_col in df.columns:
                    fig = px.scatter(df, x=x_col, y=y_col, color=color_col, title=title, color_discrete_sequence=_COLORS, opacity=0.75)
                else:
                    fig = px.scatter(df, x=x_col, y=y_col, title=title, color_discrete_sequence=_COLORS, opacity=0.75)

            elif ct in ("scatter_trend", "scattertrend", "scatterwithtrendline"):
                fig = _build_scatter_trend(df, x_col, y_col, color_col, title)

            elif ct in ("bubble", "bubblechart"):
                if size_col and size_col in df.columns:
                    fig = px.scatter(df, x=x_col, y=y_col, size=size_col,
                                     color=color_col if (color_col and color_col in df.columns) else None,
                                     title=title, color_discrete_sequence=_COLORS, size_max=50, opacity=0.75)
                else:
                    fig = px.scatter(df, x=x_col, y=y_col, title=title, color_discrete_sequence=_COLORS)

            elif ct in ("histogram", "hist", "dist"):
                fig = px.histogram(df, x=x_col, title=title, color_discrete_sequence=_COLORS, nbins=30)

            elif ct in ("box", "boxplot"):
                fig = px.box(df, x=x_col, y=y_col, title=title, color_discrete_sequence=_COLORS)

            elif ct in ("box_group", "boxgroup", "groupedbox"):
                if color_col and color_col in df.columns:
                    fig = px.box(df, x=x_col, y=y_col, color=color_col, title=title, color_discrete_sequence=_COLORS)
                else:
                    fig = px.box(df, x=x_col, y=y_col, title=title, color_discrete_sequence=_COLORS)

            elif ct in ("funnel", "funnelchart"):
                fig = px.funnel(df, y=x_col, x=y_col, title=title, color_discrete_sequence=_COLORS)

            else:
                fig = px.bar(df, x=x_col, y=y_col, title=title, color_discrete_sequence=_COLORS)

            _apply_layout(fig, title)
            return _to_json(fig)

        except Exception as exc:
            return {"error": f"Chart failed ({ct}): {exc}"}

    # ================================================================
    # Complex chart builders — call directly from analysis functions
    # ================================================================

    @classmethod
    def build_pareto(cls, df, label_col, value_col, title=None):
        try:
            fig = go.Figure()
            fig.add_trace(go.Bar(x=df[label_col], y=df[value_col], name=value_col,
                                  marker_color=_COLORS[0], opacity=0.85))
            if "Cumulative %" in df.columns:
                fig.add_trace(go.Scatter(
                    x=df[label_col], y=df["Cumulative %"], name="Cumulative %",
                    mode="lines+markers", yaxis="y2",
                    line=dict(color=_COLORS[4], width=2),
                    marker=dict(size=6, color=_COLORS[4]),
                ))
                fig.update_layout(
                    yaxis2=dict(title="Cumulative %", overlaying="y", side="right",
                                range=[0, 105], ticksuffix="%", showgrid=False))
                fig.add_hline(y=80, line_dash="dash", line_color="#DC2626",
                              annotation_text="80% threshold",
                              annotation_position="top right", yref="y2")
            fig.update_layout(title_text=title or f"Pareto: {value_col} by {label_col}")
            _apply_layout(fig)
            return _to_json(fig)
        except Exception as exc:
            return {"error": f"Pareto chart failed: {exc}"}

    @classmethod
    def build_heatmap(cls, pivot_df, title=None, fmt=".0f", colorscale="Blues"):
        try:
            fig = px.imshow(pivot_df, text_auto=fmt, aspect="auto",
                            color_continuous_scale=colorscale,
                            title=title or "Heatmap")
            fig.update_xaxes(side="bottom")
            _apply_layout(fig)
            return _to_json(fig)
        except Exception as exc:
            return {"error": f"Heatmap failed: {exc}"}

    @classmethod
    def build_treemap(cls, df, path_cols, value_col, title=None):
        try:
            valid_path = [c for c in path_cols if c in df.columns]
            if not valid_path or value_col not in df.columns:
                return {"error": "Invalid columns for treemap."}
            fig = px.treemap(df, path=valid_path, values=value_col,
                             color=value_col, color_continuous_scale="Blues",
                             title=title or f"{value_col} Treemap")
            fig.update_traces(textinfo="label+value+percent root")
            _apply_layout(fig)
            return _to_json(fig)
        except Exception as exc:
            return {"error": f"Treemap failed: {exc}"}

    @classmethod
    def build_waterfall(cls, labels, values, measures=None, title=None):
        try:
            if measures is None:
                measures = ["relative"] * len(labels)
            fig = go.Figure(go.Waterfall(
                name="", orientation="v",
                measure=measures, x=list(labels),
                y=[_safe(v) for v in values],
                connector={"line": {"color": "#94A3B8", "width": 1}},
                increasing={"marker": {"color": _COLORS[2]}},
                decreasing={"marker": {"color": _COLORS[4]}},
                totals={"marker": {"color": _COLORS[0]}},
                text=[f"{_safe(v):+,.0f}" for v in values],
                textposition="outside",
            ))
            fig.update_layout(title_text=title or "Waterfall Chart", showlegend=False)
            _apply_layout(fig)
            return _to_json(fig)
        except Exception as exc:
            return {"error": f"Waterfall failed: {exc}"}

    @classmethod
    def build_gauge(cls, value, target, label="", title=None):
        try:
            max_range = max(target * 1.5, value * 1.3, 1.0)
            pct = round(value / target * 100, 1) if target else 0
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=round(float(value), 2),
                number={"valueformat": ",.2f"},
                delta={"reference": float(target), "valueformat": ",.2f",
                       "increasing": {"color": _COLORS[2]},
                       "decreasing": {"color": _COLORS[4]}},
                gauge={
                    "axis": {"range": [0, max_range], "tickwidth": 1, "tickcolor": "#94A3B8"},
                    "bar": {"color": _COLORS[0], "thickness": 0.3},
                    "bgcolor": "white",
                    "steps": [
                        {"range": [0, target * 0.5], "color": "#FEE2E2"},
                        {"range": [target * 0.5, target * 0.9], "color": "#FEF3C7"},
                        {"range": [target * 0.9, max_range], "color": "#D1FAE5"},
                    ],
                    "threshold": {
                        "line": {"color": "#DC2626", "width": 4},
                        "thickness": 0.75,
                        "value": float(target),
                    },
                },
                title={"text": f"{label}<br><sub>{pct}% of target</sub>",
                       "font": {"size": 16, "color": "#334155"}},
            ))
            fig.update_layout(margin=dict(l=40, r=40, t=80, b=30))
            return _to_json(fig)
        except Exception as exc:
            return {"error": f"Gauge failed: {exc}"}

    @classmethod
    def build_radar(cls, categories, values, series_name="Value", title=None):
        try:
            cats = list(categories) + [categories[0]]
            vals = [_safe(v) for v in values] + [_safe(values[0])]
            fig = go.Figure(go.Scatterpolar(
                r=vals, theta=cats, fill="toself", name=series_name,
                line=dict(color=_COLORS[0], width=2),
                marker=dict(color=_COLORS[0], size=6),
                fillcolor="rgba(37,99,235,0.2)",
            ))
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, showticklabels=True, gridcolor="#E2E8F0"),
                    angularaxis=dict(gridcolor="#E2E8F0"),
                    bgcolor="rgba(0,0,0,0)",
                ),
                showlegend=False, title_text=title or "Radar Chart",
            )
            _apply_layout(fig)
            return _to_json(fig)
        except Exception as exc:
            return {"error": f"Radar failed: {exc}"}

    @classmethod
    def build_combo_bar_line(cls, df, x_col, bar_col, line_col, title=None):
        try:
            fig = go.Figure()
            fig.add_trace(go.Bar(x=df[x_col], y=df[bar_col], name=bar_col,
                                  marker_color=_COLORS[0], opacity=0.85))
            fig.add_trace(go.Scatter(x=df[x_col], y=df[line_col], name=line_col,
                                      mode="lines+markers", yaxis="y2",
                                      line=dict(color=_COLORS[1], width=2),
                                      marker=dict(size=6)))
            fig.update_layout(
                title_text=title or f"{bar_col} & {line_col}",
                yaxis=dict(title=bar_col, showgrid=True, gridcolor="#F1F5F9"),
                yaxis2=dict(title=line_col, overlaying="y", side="right", showgrid=False),
                legend=dict(orientation="h", y=1.05),
            )
            _apply_layout(fig)
            return _to_json(fig)
        except Exception as exc:
            return {"error": f"Combo chart failed: {exc}"}

    @classmethod
    def build_cohort_heatmap(cls, retention_df, title=None):
        try:
            fig = px.imshow(retention_df, text_auto=".0f", aspect="auto",
                            color_continuous_scale="RdYlGn", zmin=0, zmax=100,
                            title=title or "Cohort Retention (%)")
            _apply_layout(fig)
            return _to_json(fig)
        except Exception as exc:
            return {"error": f"Cohort heatmap failed: {exc}"}

    @classmethod
    def build_funnel(cls, df, stage_col, value_col, title=None):
        try:
            fig = px.funnel(df, y=stage_col, x=value_col,
                            title=title or "Funnel Chart",
                            color_discrete_sequence=_COLORS)
            _apply_layout(fig)
            return _to_json(fig)
        except Exception as exc:
            return {"error": f"Funnel failed: {exc}"}


def _mark_bar_values(fig):
    for trace in fig.data:
        if hasattr(trace, "x") and trace.x is not None and len(trace.x) <= 15:
            trace.update(texttemplate="%{y:,.0f}", textposition="outside")


def _build_scatter_trend(df, x_col, y_col, color_col, title):
    fig = go.Figure()
    if color_col and color_col in df.columns:
        for i, (grp, sub) in enumerate(df.groupby(color_col)):
            fig.add_trace(go.Scatter(x=sub[x_col], y=sub[y_col], mode="markers",
                                      name=str(grp),
                                      marker=dict(color=_COLORS[i % len(_COLORS)],
                                                  opacity=0.7, size=7)))
    else:
        fig.add_trace(go.Scatter(x=df[x_col], y=df[y_col], mode="markers",
                                  name="Data",
                                  marker=dict(color=_COLORS[0], opacity=0.7, size=7)))
    try:
        x_arr = pd.to_numeric(df[x_col], errors="coerce").dropna().values
        y_arr = pd.to_numeric(df[y_col], errors="coerce").dropna().values
        n = min(len(x_arr), len(y_arr))
        x_arr, y_arr = x_arr[:n], y_arr[:n]
        if n >= 2:
            z = np.polyfit(x_arr, y_arr, 1)
            p = np.poly1d(z)
            xr = np.linspace(x_arr.min(), x_arr.max(), 200)
            fig.add_trace(go.Scatter(x=xr, y=p(xr), mode="lines", name="Trend",
                                      line=dict(color=_COLORS[4], width=2, dash="dash")))
    except Exception:
        pass
    _apply_layout(fig, title)
    return fig
