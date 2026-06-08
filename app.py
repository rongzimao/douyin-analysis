# -*- coding: utf-8 -*-
"""
抖音用户行为数据分析 — 交互式可视化平台 (性能优化版)
基于 Streamlit + Plotly

优化策略:
  1. Parquet 格式加速数据加载
  2. @st.cache_data 缓存所有聚合计算 (ttl + max_entries)
  3. 筛选器在预计算聚合结果上过滤，避免重复聚合
  4. Plotly scattergl / 合理 nbins / 精简 update_layout
  5. st.session_state 惰性渲染，仅当前标签页创建图表
  6. st.status 改善加载体验
  7. 所有 groupby 聚合封装在缓存函数中
  8. CSV → Parquet 自动转换

运行方式: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import requests
import json
import time
import warnings

warnings.filterwarnings('ignore')

# 全局 plotly 模板 — 无边框 + 网格对齐
# 配色：10色调色板
PLOTLY_COLORS = ["#6366f1", "#8b5cf6", "#10b981", "#f59e0b", "#ef4444",
                 "#3b82f6", "#ec4899", "#06b6d4", "#84cc16", "#f97316"]
# 统一字号
FONT_TITLE = 14; FONT_AXIS = 10; FONT_TICK = 10; FONT_LEGEND = 10

PLOTLY_TEMPLATE = {
    "layout": {
        "font": {"family": "Inter, sans-serif", "color": "#64748b", "size": FONT_TICK},
        "title": {"font": {"size": FONT_TITLE, "color": "#1e293b",
                            "family": "Inter, sans-serif"},
                   "x": 0, "xanchor": "left"},
        "xaxis": {
            "title": {"font": {"size": FONT_AXIS, "color": "#94a3b8"}},
            "tickfont": {"size": FONT_TICK, "color": "#94a3b8"},
            "gridcolor": "#f8fafc", "zerolinecolor": "#f1f5f9",
            "showline": True, "linewidth": 1, "linecolor": "#e2e8f0",
            "mirror": False, "showgrid": True,
        },
        "yaxis": {
            "title": {"font": {"size": FONT_AXIS, "color": "#94a3b8"}},
            "tickfont": {"size": FONT_TICK, "color": "#94a3b8"},
            "gridcolor": "#f8fafc", "zerolinecolor": "#f1f5f9",
            "showline": True, "linewidth": 1, "linecolor": "#e2e8f0",
            "mirror": False, "showgrid": True,
        },
        "plot_bgcolor": "#ffffff",
        "paper_bgcolor": "#ffffff",
        "margin": {"t": 42, "r": 16, "b": 28, "l": 16},
        "legend": {
            "font": {"size": FONT_LEGEND, "color": "#64748b"},
            "bgcolor": "rgba(255,255,255,0.6)",
            "bordercolor": "rgba(0,0,0,0)", "borderwidth": 0,
            "orientation": "h", "yanchor": "top", "y": 1.08,
            "xanchor": "left", "x": 0,
        },
        "colorway": PLOTLY_COLORS,
        "barmode": "group", "bargap": 0.18, "bargroupgap": 0.1,
        "hovermode": "x unified",
        "hoverlabel": {"bgcolor": "#1e293b", "font": {"size": 11, "color": "#fff"},
                        "bordercolor": "#1e293b", "borderradius": 6},
    }
}
# 图表统一高度常量
CHART_H_FULL = 420   # 全宽图表
CHART_H_HALF = 370   # 两列布局中的图表
CHART_H_SMALL = 320  # 饼图/小提琴等小图表
CHART_MARGIN = dict(t=42, r=16, b=28, l=16)

# ============================================================
# 页面配置
# ============================================================
st.set_page_config(
    page_title="抖音用户行为数据分析",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# 全局样式注入
# ============================================================
st.markdown("""
<style>
/* ================================================================
   抖音用户行为数据分析平台 — 现代简约仪表盘样式
   主色调：低饱和蓝紫渐变 (#6366f1 → #8b5cf6)
   ================================================================ */

/* ---- 1. 全局基础 ---- */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --color-primary: #6366f1;
    --color-primary-light: #eef2ff;
    --color-primary-dark: #4f46e5;
    --color-accent: #8b5cf6;
    --color-surface: #ffffff;
    --color-bg: #f8fafc;
    --color-text: #1e293b;
    --color-text-secondary: #64748b;
    --color-text-muted: #94a3b8;
    --color-border: #e2e8f0;
    --color-border-light: #f1f5f9;
    --shadow-sm: 0 1px 2px rgba(0,0,0,0.04);
    --shadow-md: 0 4px 16px rgba(0,0,0,0.06);
    --shadow-lg: 0 8px 24px rgba(0,0,0,0.08);
    --radius-sm: 8px;
    --radius-md: 12px;
    --radius-lg: 16px;
}

html, body, [class*="st-"], .stApp {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
    color: var(--color-text-secondary);
    line-height: 1.55;
    -webkit-font-smoothing: antialiased;
}
.stApp { background-color: var(--color-bg); }

/* 主内容区：1200px 居中 + 网格对齐 */
section.main > div[data-testid="stVerticalBlock"] {
    max-width: 1200px !important;
    margin: 0 auto !important;
    padding: 0 1.5rem !important;
}

/* ---- 2. 左侧导航栏 ---- */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #fafbff 0%, #ffffff 100%);
    border-right: 1px solid var(--color-border);
}
[data-testid="stSidebar"] > div[data-testid="stVerticalBlock"] {
    padding: 0.75rem 1rem !important;
}
[data-testid="stSidebar"] h2 {
    font-size: 1.05rem !important; font-weight: 700 !important;
    color: var(--color-text) !important; margin-bottom: 0.5rem !important;
}
[data-testid="stSidebar"] h3 {
    font-size: 0.75rem !important; font-weight: 600 !important;
    color: var(--color-text-muted) !important; text-transform: uppercase;
    letter-spacing: 0.06em; margin-bottom: 0.5rem !important;
    margin-top: 1.25rem !important;
}
[data-testid="stSidebar"] hr {
    margin: 0.75rem 0; border-color: var(--color-border);
}
[data-testid="stSidebar"] span[data-baseweb="tag"] {
    border-radius: 6px !important; font-size: 0.72rem !important;
    background: var(--color-primary-light) !important;
    color: var(--color-primary) !important;
}

/* ---- 3. KPI 指标卡片 — 网格对齐、统一尺寸 ---- */
div[data-testid="stMetric"] {
    background: var(--color-surface);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-md);
    padding: 1.25rem 0.75rem !important;
    text-align: center !important;
    box-shadow: var(--shadow-sm);
    min-height: 108px !important;
    height: 100% !important;
    display: flex !important;
    flex-direction: column !important;
    justify-content: center !important;
    align-items: center !important;
    transition: all 0.2s ease;
}
div[data-testid="stMetric"]:hover {
    box-shadow: var(--shadow-md);
    border-color: #c7d2fe;
    transform: translateY(-1px);
}
div[data-testid="stMetric"] label {
    font-size: 0.7rem !important; font-weight: 600 !important;
    color: var(--color-text-muted) !important; text-transform: uppercase;
    letter-spacing: 0.05em; margin-bottom: 6px !important;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    font-size: 1.55rem !important; font-weight: 700 !important;
    color: var(--color-text) !important; line-height: 1.15 !important;
    background: linear-gradient(135deg, var(--color-text) 0%, #334155 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

/* ---- 4. 图表列卡片 ---- */
div[data-testid="stHorizontalBlock"] {
    gap: 20px !important;
}
/* 每列作为卡片：白底+圆角+阴影+内边距 */
div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {
    background: #ffffff;
    border-radius: 12px !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08) !important;
    padding: 20px !important;
    overflow: hidden;
}
/* 卡片内标题 */
div[data-testid="column"] h3, div[data-testid="column"] .card-title {
    font-size: 16px !important;
    font-weight: 700 !important;
    color: #333333 !important;
    margin: 0 0 12px 0 !important;
    text-align: left !important;
}
/* 卡片内图表撑满 */
div[data-testid="column"] [data-testid="stPlotlyChart"] {
    margin-bottom: 0 !important;
    border: none !important;
    box-shadow: none !important;
    border-radius: 0 !important;
}

/* ---- 5. Radio 标签页 — 选中态用主色渐变 ---- */
div[role="radiogroup"] {
    background: var(--color-surface);
    border: 1px solid var(--color-border);
    border-radius: var(--radius-md);
    padding: 4px !important;
    gap: 2px;
    margin-bottom: 1.25rem !important;
}
div[role="radiogroup"] label {
    border-radius: 10px !important;
    padding: 0.45rem 1.25rem !important;
    font-weight: 500 !important; font-size: 0.85rem !important;
    color: var(--color-text-secondary) !important;
    transition: all 0.2s ease; margin: 0 !important;
}
div[role="radiogroup"] label:hover {
    color: var(--color-primary) !important;
    background: var(--color-primary-light) !important;
}
div[role="radiogroup"] label[data-selected="true"],
div[role="radiogroup"] label[aria-checked="true"] {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
    color: #ffffff !important;
    box-shadow: 0 2px 8px rgba(99,102,241,0.3);
}

/* ---- 6. 按钮 — 主色渐变 ---- */
div.stButton > button {
    border-radius: var(--radius-sm) !important;
    border: 1px solid var(--color-border) !important;
    background: var(--color-surface) !important;
    color: var(--color-text-secondary) !important;
    font-weight: 500 !important; padding: 0.45rem 1.15rem !important;
    transition: all 0.2s ease;
}
div.stButton > button:hover {
    background: var(--color-primary-light) !important;
    border-color: #c7d2fe !important;
    color: var(--color-primary) !important;
}
div.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
    color: #ffffff !important; border: none !important;
    box-shadow: 0 2px 8px rgba(99,102,241,0.25);
}
div.stButton > button[kind="primary"]:hover {
    box-shadow: 0 4px 16px rgba(99,102,241,0.4);
    transform: translateY(-1px);
}

/* ---- 7. 图表容器 — 严格网格对齐、无边框 ---- */
[data-testid="stPlotlyChart"] {
    margin-bottom: 0.75rem !important;
    border-radius: 10px !important;
    overflow: hidden !important;
    background: #ffffff !important;
    border: 1px solid #f1f5f9 !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.04) !important;
    padding: 0 !important;
}
/* 同一行图表容器高度强制一致 */
div[data-testid="stHorizontalBlock"] > div[data-testid="stVerticalBlock"] {
    align-items: stretch !important;
}
div[data-testid="stHorizontalBlock"] > div[data-testid="stVerticalBlock"] > div {
    flex: 1 !important;
}

/* ---- 8. 分隔线 ---- */
.stMain hr, section.main hr, hr {
    border: none !important;
    border-top: 1px solid var(--color-border) !important;
    margin: 1.25rem 0 !important;
}

/* ---- 9. 标题字号统一 (标题14px/二级18px) ---- */
section.main h3 {
    font-size: 14px !important; font-weight: 600 !important;
    color: #1e293b !important;
    margin: 0 0 6px 0 !important;
}
section.main h2 {
    font-size: 18px !important; font-weight: 700 !important;
    color: #1e293b !important;
    margin: 0 0 8px 0 !important;
}

/* ---- 10. Sidebar 输入控件 ---- */
[data-testid="stTextArea"] textarea {
    background: #f8fafc !important; border: 1px solid var(--color-border) !important;
    border-radius: var(--radius-sm) !important; font-size: 0.84rem !important;
}
[data-testid="stTextArea"] textarea:focus {
    border-color: var(--color-primary) !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.1) !important;
}
[data-testid="stMultiSelect"] div[data-baseweb="select"] {
    border-radius: var(--radius-sm) !important;
}
div[data-testid="stSlider"] div[data-testid="stThumbValue"] {
    background: var(--color-primary) !important; color: #ffffff !important;
}
/* Date input */
div[data-testid="stDateInput"] input {
    border-radius: var(--radius-sm) !important;
}

/* ---- 11. 隐藏 Streamlit 默认装饰 ---- */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stToolbar"] { display: none !important; }
[data-testid="stDecoration"] { display: none !important; }
.stDeployButton { display: none !important; }
div[data-testid="stStatusWidget"] { display: none !important; }

/* ---- 12. Sidebar Metric (筛选概览数字) ---- */
[data-testid="stSidebar"] [data-testid="stMetric"] {
    min-height: unset !important; padding: 0.6rem 0.5rem !important;
    border-radius: var(--radius-sm) !important;
    box-shadow: none !important;
}
[data-testid="stSidebar"] [data-testid="stMetric"] [data-testid="stMetricValue"] {
    font-size: 1.2rem !important;
    -webkit-text-fill-color: var(--color-primary) !important;
}

/* ---- 13. Expander / 历史问答 ---- */
[data-testid="stExpander"] {
    border: 1px solid var(--color-border) !important;
    border-radius: var(--radius-md) !important;
}
[data-testid="stStatus"] {
    border-radius: var(--radius-md) !important;
    border: 1px solid var(--color-border) !important;
}

/* ---- 14. 滚动条 ---- */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #94a3b8; }

/* ---- 15. 全局下拉/输入控件 ---- */
div[data-baseweb="select"] { border-radius: var(--radius-sm) !important; }
input[data-baseweb="input"] { border-radius: var(--radius-sm) !important; }

</style>
""", unsafe_allow_html=True)

# ============================================================
# Session State 初始化 — 惰性渲染 & 数据就绪标记
# ============================================================
DEFAULT_TABS = ["👤 用户分析", "✍️ 作者分析", "🎬 作品分析", "📊 综合仪表盘"]

if "active_tab" not in st.session_state:
    st.session_state.active_tab = DEFAULT_TABS[0]
if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False
if "load_start_time" not in st.session_state:
    st.session_state.load_start_time = None


# ============================================================
# 1. 数据加载 — CSV → Parquet 自动转换 + 缓存
# ============================================================
@st.cache_data(ttl=3600, max_entries=1)
def _load_raw_df_from_parquet(parquet_path: str) -> pd.DataFrame:
    """从 Parquet 读取原始数据 (内部函数)"""
    return pd.read_parquet(parquet_path)


@st.cache_data(ttl=3600, max_entries=1, show_spinner=False)
def load_data() -> pd.DataFrame:
    """
    加载全量数据。
    首次运行自动将 CSV 转为 Parquet，后续读取 Parquet (快 5-10x)。
    """
    csv_path = "douyin_dataset (1).csv"
    parquet_path = "douyin_dataset.parquet"

    if not os.path.exists(parquet_path):
        st.info("🔄 首次运行，正在将 CSV 转换为 Parquet 格式以加速后续加载...")
        t0 = time.time()
        df = pd.read_csv(csv_path)
        df.to_parquet(parquet_path, index=False, compression="snappy")
        elapsed = time.time() - t0
        st.success(f"✅ 转换完成! 耗时 {elapsed:.1f}s, 文件大小: {os.path.getsize(parquet_path)/1024**2:.1f} MB")
    else:
        df = _load_raw_df_from_parquet(parquet_path)

    # 清理与类型转换
    if "Unnamed: 0" in df.columns:
        df.drop(columns=["Unnamed: 0"], inplace=True)

    df["real_time"] = pd.to_datetime(df["real_time"])
    df["date"] = pd.to_datetime(df["date"])
    df["day_of_week"] = df["date"].dt.dayofweek
    df["weekday_name"] = df["date"].dt.day_name()
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["user_city"] = df["user_city"].astype(int)

    return df


# ============================================================
# 2. 全量聚合计算 — 全部 @st.cache_data，只计算一次
# ============================================================
@st.cache_data(ttl=3600, max_entries=12, show_spinner=False)
def compute_daily_stats(df: pd.DataFrame) -> pd.DataFrame:
    """每日聚合统计 (全量)"""
    return df.groupby("date").agg(
        total_records=("uid", "count"),
        unique_users=("uid", "nunique"),
        unique_items=("item_id", "nunique"),
        unique_authors=("author_id", "nunique"),
        avg_finish=("finish", "mean"),
        avg_like=("like", "mean"),
    ).reset_index()


@st.cache_data(ttl=3600, max_entries=12, show_spinner=False)
def compute_hourly_stats(df: pd.DataFrame) -> pd.DataFrame:
    """小时聚合统计 (全量)"""
    hourly = df.groupby("H").agg(
        record_count=("uid", "count"),
        unique_users=("uid", "nunique"),
        avg_finish=("finish", "mean"),
        avg_like=("like", "mean"),
    ).reset_index()
    return hourly


@st.cache_data(ttl=3600, max_entries=12, show_spinner=False)
def compute_city_stats(df: pd.DataFrame) -> pd.DataFrame:
    """城市聚合统计 (全量)"""
    return df.groupby("user_city").agg(
        record_count=("uid", "count"),
        unique_users=("uid", "nunique"),
        avg_finish=("finish", "mean"),
        avg_like=("like", "mean"),
    ).reset_index().sort_values("record_count", ascending=False)


@st.cache_data(ttl=3600, max_entries=12, show_spinner=False)
def compute_channel_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Channel 聚合统计 (全量)"""
    return df.groupby("channel").agg(
        record_count=("uid", "count"),
        avg_finish=("finish", "mean"),
        avg_like=("like", "mean"),
    ).reset_index()


@st.cache_data(ttl=3600, max_entries=12, show_spinner=False)
def compute_user_stats(df: pd.DataFrame) -> pd.DataFrame:
    """用户级聚合 (全量)"""
    return df.groupby("uid").agg(
        watch_count=("item_id", "count"),
        item_count=("item_id", "nunique"),
        finish_rate=("finish", "mean"),
        like_rate=("like", "mean"),
        avg_duration=("duration_time", "mean"),
        user_city=("user_city", "first"),
    ).reset_index()


@st.cache_data(ttl=3600, max_entries=12, show_spinner=False)
def compute_author_stats(df: pd.DataFrame) -> pd.DataFrame:
    """作者级聚合 (全量)"""
    return df.groupby("author_id").agg(
        video_count=("item_id", "nunique"),
        total_views=("uid", "count"),
        avg_duration=("duration_time", "mean"),
        avg_finish_rate=("finish", "mean"),
        avg_like_rate=("like", "mean"),
        total_likes=("like", "sum"),
    ).reset_index()


@st.cache_data(ttl=3600, max_entries=12, show_spinner=False)
def compute_item_stats(df: pd.DataFrame) -> pd.DataFrame:
    """作品级聚合 (全量)"""
    return df.groupby("item_id").agg(
        view_count=("uid", "count"),
        total_likes=("like", "sum"),
        like_rate=("like", "mean"),
        finish_rate=("finish", "mean"),
        avg_duration=("duration_time", "mean"),
        author_id=("author_id", "first"),
    ).reset_index()


@st.cache_data(ttl=3600, max_entries=12, show_spinner=False)
def compute_hourly_weekday_heatmap(df: pd.DataFrame) -> pd.DataFrame:
    """时段×星期 透视表 (全量)"""
    df_temp = df.copy()
    weekday_map = {
        "Monday": "周一", "Tuesday": "周二", "Wednesday": "周三",
        "Thursday": "周四", "Friday": "周五", "Saturday": "周六", "Sunday": "周日",
    }
    df_temp["weekday_label"] = df_temp["date"].dt.day_name().map(weekday_map)
    pivot = df_temp.pivot_table(
        index="H", columns="weekday_label", values="uid", aggfunc="count"
    )
    weekday_order = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
    return pivot.reindex(columns=[w for w in weekday_order if w in pivot.columns])


# ============================================================
# 3. 数据过滤 — 在预计算聚合结果上筛选，避免重新聚合
# ============================================================
def filter_daily_stats(daily_stats: pd.DataFrame, date_range) -> pd.DataFrame:
    """从预计算的 daily_stats 中按日期范围筛选"""
    start, end = date_range
    return daily_stats[
        (daily_stats["date"].dt.date >= start) & (daily_stats["date"].dt.date <= end)
    ]


def filter_city_stats(city_stats: pd.DataFrame, selected_cities: list) -> pd.DataFrame:
    """从预计算的 city_stats 中按城市筛选"""
    if not selected_cities:
        return city_stats.head(0)
    return city_stats[city_stats["user_city"].isin(selected_cities)]


def filter_channel_stats(channel_stats: pd.DataFrame, selected_channels: list) -> pd.DataFrame:
    """从预计算的 channel_stats 中按频道筛选"""
    if not selected_channels:
        return channel_stats.head(0)
    return channel_stats[channel_stats["channel"].isin(selected_channels)]


def filter_user_stats(user_stats: pd.DataFrame, selected_cities: list) -> pd.DataFrame:
    """从预计算的 user_stats 中按城市筛选"""
    if not selected_cities:
        return user_stats.head(0)
    return user_stats[user_stats["user_city"].isin(selected_cities)]


@st.cache_data(ttl=600, max_entries=32, show_spinner=False)
def filter_raw_df(
    date_start, date_end, selected_cities: tuple, selected_channels: tuple
) -> pd.DataFrame:
    """
    筛选原始数据 — 带缓存。
    内部调用 load_data() 获取 df，避免 df 作为缓存键导致 id 变化。
    """
    df = load_data()
    mask = (
        (df["date"].dt.date >= date_start)
        & (df["date"].dt.date <= date_end)
        & (df["user_city"].isin(list(selected_cities)))
        & (df["channel"].isin(list(selected_channels)))
    )
    return df[mask]


@st.cache_data(ttl=600, max_entries=32, show_spinner=False)
def compute_filtered_summary(
    date_start, date_end, selected_cities: tuple, selected_channels: tuple
) -> dict:
    """
    一次性计算 filtered_df 的所有常用统计指标并缓存。
    仅以筛选参数为缓存键，df 从 load_data() 内部获取。
    """
    fdf = filter_raw_df(date_start, date_end, selected_cities, selected_channels)
    n = len(fdf)
    n_users = max(fdf["uid"].nunique(), 1)
    n_authors = fdf["author_id"].nunique()
    n_items = fdf["item_id"].nunique()
    n_cities = fdf["user_city"].nunique()
    n_channels = fdf["channel"].nunique()
    min_date = fdf["date"].min().date()
    max_date = fdf["date"].max().date()

    weekend_mask = fdf["is_weekend"] == 1
    weekday_mask = ~weekend_mask

    # 人均数据
    avg_watch_per_user = n / n_users
    avg_items_per_user = fdf.groupby("uid")["item_id"].nunique().mean()
    avg_duration = fdf["duration_time"].mean()
    avg_items_per_author = fdf.groupby("author_id")["item_id"].nunique().mean()

    # 完成率和点赞率
    avg_finish = fdf["finish"].mean()
    avg_like = fdf["like"].mean()
    median_duration = fdf["duration_time"].median()

    # 周末/工作日
    weekend_finish = fdf.loc[weekend_mask, "finish"].mean() if weekend_mask.any() else float("nan")
    weekday_finish = fdf.loc[weekday_mask, "finish"].mean() if weekday_mask.any() else float("nan")
    n_weekend = weekend_mask.sum()
    n_weekday = weekday_mask.sum()

    # 活跃时段
    hourly_counts = fdf.groupby("H").size()
    peak_hour = int(hourly_counts.idxmax())
    peak_hour_count = int(hourly_counts.max())
    early_finish = fdf.loc[fdf["H"].between(0, 6), "finish"].mean()
    prime_finish = fdf.loc[fdf["H"].between(18, 23), "finish"].mean()

    # 仪表盘专用聚合 — Top10 城市
    top10_cities = (
        fdf["user_city"].value_counts().nlargest(10)
        .reset_index()
    )
    top10_cities.columns = ["city", "count"]
    top10_cities["city"] = top10_cities["city"].astype(str)

    # 仪表盘专用聚合 — Channel 分布
    channel_dist = (
        fdf.groupby("channel").size().reset_index(name="count")
    )
    channel_dist["channel"] = channel_dist["channel"].apply(lambda x: f"Ch-{x}")

    # 仪表盘专用聚合 — 时段分布 (H)
    hourly_dash = (
        fdf.groupby("H")
        .agg(记录数=("uid", "count"), 完成率=("finish", "mean"), 点赞率=("like", "mean"))
        .reset_index()
    )

    return {
        "len": n, "n_users": n_users, "n_authors": n_authors,
        "n_items": n_items, "n_cities": n_cities, "n_channels": n_channels,
        "min_date": min_date, "max_date": max_date,
        "avg_finish": avg_finish, "avg_like": avg_like,
        "avg_duration": avg_duration, "median_duration": median_duration,
        "avg_watch_per_user": avg_watch_per_user,
        "avg_items_per_user": avg_items_per_user,
        "avg_items_per_author": avg_items_per_author,
        "weekend_finish": weekend_finish, "weekday_finish": weekday_finish,
        "n_weekend": n_weekend, "n_weekday": n_weekday,
        "peak_hour": peak_hour, "peak_hour_count": peak_hour_count,
        "early_finish": early_finish, "prime_finish": prime_finish,
        "top10_cities": top10_cities,
        "channel_dist": channel_dist,
        "hourly_dash": hourly_dash,
    }


# ============================================================
# 4. 图表构建函数 — 按标签页分组，仅在需要时调用
# ============================================================

# ---- 4a. 用户分析图表 ----
def render_user_analysis(
    filtered_df, user_stats_f, city_stats_f, hourly_stats, heatmap_data,
):
    """渲染用户分析标签页的所有图表"""
    st.markdown("## 👤 用户维度分析")

    # Row 1: 用户观看次数分布 + 用户活跃度分层
    r1l, r1r = st.columns(2, gap="medium")
    with r1l:
        st.markdown("### 用户观看次数分布")
        bin_option = st.radio(
            "X轴刻度", ["线性", "对数"], key="user_watch_scale", horizontal=True
        )
        log_x = bin_option == "对数"
        fig = px.histogram(
            user_stats_f, x="watch_count", nbins=50,
            log_x=log_x, log_y=True,
            color_discrete_sequence=[PLOTLY_COLORS[5]],  # blue
            title="用户观看次数分布 (Y轴对数)",
            labels={"watch_count": "观看次数"},
        )
        fig.update_layout(bargap=0.05, height=CHART_H_HALF,
                          plot_bgcolor="white", paper_bgcolor="white",
                          xaxis_showline=False, yaxis_showline=False,
                          legend_borderwidth=0,
                          margin=dict(l=20, r=20, t=20, b=20))
        median_val = user_stats_f["watch_count"].median()
        fig.add_vline(x=median_val, line_dash="dash", line_color="#ef4444",
                      annotation_text=f"中位数: {median_val:.0f}")
        st.plotly_chart(fig, use_container_width=True)

    with r1r:
        st.markdown("### 用户活跃度分层")
        bins_edges = [0, 2, 5, 10, 30, 100, float("inf")]
        bins_labels = ["1次", "2-5次", "6-10次", "11-30次", "31-100次", ">100次"]
        level_series = pd.cut(
            user_stats_f["watch_count"], bins=bins_edges, labels=bins_labels
        )
        level_counts = level_series.value_counts().reset_index()
        level_counts.columns = ["活跃度", "用户数"]
        fig = px.pie(
            level_counts, values="用户数", names="活跃度",
            color_discrete_sequence=PLOTLY_COLORS,
            title="用户活跃度分层 (按观看次数)",
        )
        fig.update_traces(textposition="inside", textinfo="percent+value")
        fig.update_layout(height=CHART_H_HALF,
                          plot_bgcolor="white", paper_bgcolor="white",
                          xaxis_showline=False, yaxis_showline=False,
                          legend_borderwidth=0,
                          margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

    # Row 2 [Group 1]: 用户城市分布 Top20 + 用户完成率 vs 点赞率 — 严格对齐
    st.markdown("---")
    g1l, g1r = st.columns(2, gap="medium")
    with g1l:
        st.markdown(
            "<h3 style='font-size:16px;font-weight:700;color:#333;margin:0 0 12px 0;'>"
            "用户城市分布 Top20</h3>", unsafe_allow_html=True)
        top20_cities = city_stats_f.head(20).copy()
        top20_cities["city_label"] = top20_cities["user_city"].astype(str)
        fig = px.bar(
            top20_cities, x="record_count", y="city_label", orientation="h",
            color="record_count", color_continuous_scale="Blues",
            labels={"record_count": "记录数", "city_label": "城市ID"},
        )
        fig.update_layout(
            yaxis=dict(categoryorder="total ascending"),
            height=CHART_H_FULL,
            plot_bgcolor="white", paper_bgcolor="white",
            xaxis_showline=False, yaxis_showline=False,
            legend_borderwidth=0,
            margin=dict(l=20, r=20, t=20, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

    with g1r:
        st.markdown(
            "<h3 style='font-size:16px;font-weight:700;color:#333;margin:0 0 12px 0;'>"
            "用户完成率 vs 点赞率</h3>", unsafe_allow_html=True)
        fig = px.scatter(
            user_stats_f, x="finish_rate", y="like_rate",
            size=np.log1p(user_stats_f["watch_count"]), size_max=15,
            color="watch_count", color_continuous_scale="Viridis",
            opacity=0.5,
            title=f"用户完成率 vs 点赞率 ({len(user_stats_f):,} 用户)",
            labels={
                "finish_rate": "完成率", "like_rate": "点赞率",
                "watch_count": "观看次数",
            },
            render_mode="webgl",
        )
        fig.update_layout(
            height=CHART_H_FULL,
            plot_bgcolor="white", paper_bgcolor="white",
            xaxis_showline=False, yaxis_showline=False,
            legend_borderwidth=0,
            margin=dict(l=20, r=20, t=20, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

    # --- 24小时时段偏好 ---
    st.markdown("### 用户观看时段偏好 (24小时)")
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(
            x=hourly_stats["H"], y=hourly_stats["record_count"],
            name="记录数", marker_color=PLOTLY_COLORS[0], opacity=0.8,
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=hourly_stats["H"], y=hourly_stats["avg_finish"] * 100,
            name="完成率(%)", mode="lines+markers",
            line=dict(color=PLOTLY_COLORS[2], width=2),
        ),
        secondary_y=True,
    )
    fig.add_trace(
        go.Scatter(
            x=hourly_stats["H"], y=hourly_stats["avg_like"] * 100,
            name="点赞率(%)", mode="lines+markers",
            line=dict(color=PLOTLY_COLORS[4], width=2),
        ),
        secondary_y=True,
    )
    fig.update_layout(
        title="24小时活跃度与行为趋势",
        xaxis=dict(title="小时", dtick=2),
        height=CHART_H_HALF, margin=CHART_MARGIN,
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="记录数", secondary_y=False)
    fig.update_yaxes(title_text="比率 (%)", secondary_y=True)
    st.plotly_chart(fig, use_container_width=True)


# ---- 4b. 作者分析图表 ----
def render_author_analysis(author_stats_f, filtered_df):
    """渲染作者分析标签页的所有图表"""
    st.markdown("## ✍️ 作者维度分析")

    # Row 1: 作者作品数分布 + 作者平均作品时长分布
    r1l, r1r = st.columns(2, gap="medium")
    with r1l:
        st.markdown("### 作者发布作品数分布")
        log_author_x = st.checkbox("X轴对数", value=True, key="author_log")
        fig = px.histogram(
            author_stats_f, x="video_count", nbins=50,
            log_x=log_author_x, log_y=True,
            color_discrete_sequence=[PLOTLY_COLORS[1]],  # purple
            title="作者发布作品数分布",
            labels={"video_count": "作品数"},
        )
        fig.update_layout(bargap=0.05, height=CHART_H_HALF,
                          plot_bgcolor="white", paper_bgcolor="white",
                          xaxis_showline=False, yaxis_showline=False,
                          legend_borderwidth=0,
                          margin=dict(l=20, r=20, t=20, b=20))
        fig.add_vline(x=author_stats_f["video_count"].median(), line_dash="dash",
                      line_color="#ef4444",
                      annotation_text=f"中位数: {author_stats_f['video_count'].median():.0f}")
        st.plotly_chart(fig, use_container_width=True)

    with r1r:
        st.markdown("### 作者平均作品时长分布")
        duration_max = st.slider("时长上限(秒)", 10, 120, 60, 10, key="author_dur_max")
        dur_clipped = author_stats_f[author_stats_f["avg_duration"] <= duration_max]
        fig = px.histogram(
            dur_clipped, x="avg_duration", nbins=50,
            color_discrete_sequence=["#8b5cf6"],
            title=f"作者平均作品时长分布 (≤{duration_max}s)",
            labels={"avg_duration": "平均时长 (秒)"},
        )
        fig.update_layout(bargap=0.05, height=CHART_H_HALF,
                          plot_bgcolor="white", paper_bgcolor="white",
                          xaxis_showline=False, yaxis_showline=False,
                          legend_borderwidth=0,
                          margin=dict(l=20, r=20, t=20, b=20))
        fig.add_vline(x=author_stats_f["avg_duration"].median(), line_dash="dash",
                      line_color="#ef4444",
                      annotation_text=f"中位数: {author_stats_f['avg_duration'].median():.1f}s")
        st.plotly_chart(fig, use_container_width=True)

    # Row 2 [Group 2]: 最活跃作者 + 最受欢迎作者 — 严格对齐
    st.markdown("---")
    g2l, g2r = st.columns(2, gap="medium")
    with g2l:
        st.markdown(
            "<h3 style='font-size:16px;font-weight:700;color:#333;margin:0 0 12px 0;'>"
            "最活跃作者 Top15</h3>", unsafe_allow_html=True)
        top_auth = author_stats_f.nlargest(15, "video_count")
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=[f"A-{int(a)}" for a in top_auth["author_id"]],
            x=top_auth["video_count"], orientation="h",
            marker=dict(color=top_auth["video_count"],
                        colorscale=["#c7d2fe", "#6366f1"], showscale=True),
            name="作品数",
        ))
        fig.update_layout(
            height=CHART_H_FULL, xaxis_title="作品数",
            plot_bgcolor="white", paper_bgcolor="white",
            xaxis_showline=False, yaxis_showline=False,
            legend_borderwidth=0,
            margin=dict(l=20, r=20, t=20, b=20),
            yaxis=dict(categoryorder="total ascending"),
        )
        st.plotly_chart(fig, use_container_width=True)

    with g2r:
        st.markdown(
            "<h3 style='font-size:16px;font-weight:700;color:#333;margin:0 0 12px 0;'>"
            "最受欢迎作者 Top15（按点赞数）</h3>", unsafe_allow_html=True)
        top_likes = author_stats_f.nlargest(15, "total_likes")
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=[f"A-{int(a)}" for a in top_likes["author_id"]],
            x=top_likes["total_likes"], orientation="h",
            marker=dict(color=top_likes["total_likes"],
                        colorscale=["#fecaca", "#ef4444"], showscale=True),
            name="总点赞数",
            hovertemplate="作者: %{y}<br>点赞: %{x}<br>观看: %{customdata[0]:,}<br>作品: %{customdata[1]}",
            customdata=top_likes[["total_views", "video_count"]],
        ))
        fig.update_layout(
            height=CHART_H_FULL, xaxis_title="总点赞数",
            plot_bgcolor="white", paper_bgcolor="white",
            xaxis_showline=False, yaxis_showline=False,
            legend_borderwidth=0,
            margin=dict(l=20, r=20, t=20, b=20),
            yaxis=dict(categoryorder="total ascending"),
        )
        st.plotly_chart(fig, use_container_width=True)

    # --- 作者生态热力图 (scattergl) ---
    st.markdown("### 作者生态: 作品数 vs 总观看数")
    eco = author_stats_f[author_stats_f["video_count"] <= 50]
    fig = px.density_heatmap(
        eco, x="video_count", y="total_views",
        nbinsx=50, nbinsy=50,
        color_continuous_scale="YlOrRd",
        title="作者作品数 vs 总观看数 密度热力图 (作品数≤50)",
        labels={"video_count": "作品数", "total_views": "总观看数"},
    )
    fig.update_layout(height=CHART_H_FULL, margin=CHART_MARGIN)
    st.plotly_chart(fig, use_container_width=True)


# ---- 4c. 作品分析图表 ----
def render_item_analysis(filtered_df, daily_stats_f, heatmap_data):
    """渲染作品分析标签页的所有图表"""
    st.markdown("## 🎬 作品维度分析")

    # Row 1: 视频时长分布 + 每日趋势
    r1l, r1r = st.columns(2, gap="medium")
    with r1l:
        st.markdown("### 视频时长分布")
        dur_upper = st.slider("时长上限(秒)", 10, 300, 60, 10, key="item_dur_slider")
        dur_data = filtered_df["duration_time"].clip(0, dur_upper)
        fig = px.histogram(
            x=dur_data, nbins=min(dur_upper, 60),
            color_discrete_sequence=[PLOTLY_COLORS[7]],  # cyan
            title=f"视频时长分布 (≤{dur_upper}s)",
            labels={"x": "时长 (秒)", "y": "记录数"},
        )
        fig.update_layout(bargap=0.05, height=CHART_H_HALF,
                          plot_bgcolor="white", paper_bgcolor="white",
                          xaxis_showline=False, yaxis_showline=False,
                          legend_borderwidth=0,
                          margin=dict(l=20, r=20, t=20, b=20))
        fig.add_vline(x=filtered_df["duration_time"].median(), line_dash="dash",
                      line_color="#ef4444",
                      annotation_text=f"中位数: {filtered_df['duration_time'].median():.0f}s")
        fig.add_vline(x=filtered_df["duration_time"].mean(), line_dash="dot",
                      line_color="#f59e0b",
                      annotation_text=f"均值: {filtered_df['duration_time'].mean():.1f}s")
        st.plotly_chart(fig, use_container_width=True)

    with r1r:
        st.markdown("### 每日作品/记录数趋势")
        trend_metric = st.radio("显示指标", ["总记录数", "去重作品数", "两者"],
                                horizontal=True, key="trend_metric")
        fig = go.Figure()
        if trend_metric in ["总记录数", "两者"]:
            fig.add_trace(go.Scatter(
                x=daily_stats_f["date"], y=daily_stats_f["total_records"],
                mode="lines", name="总记录数",
                line=dict(color="#6366f1", width=2),
                fill="tozeroy", fillcolor="rgba(99,102,241,0.12)",
            ))
        if trend_metric in ["去重作品数", "两者"]:
            fig.add_trace(go.Scatter(
                x=daily_stats_f["date"], y=daily_stats_f["unique_items"],
                mode="lines", name="去重作品数",
                line=dict(color="#10b981", width=2),
                fill="tozeroy", fillcolor="rgba(16,185,129,0.10)",
            ))
        fig.update_layout(
            title="每日趋势", xaxis_title="日期", yaxis_title="数量",
            height=CHART_H_HALF, hovermode="x unified",
            plot_bgcolor="white", paper_bgcolor="white",
            xaxis_showline=False, yaxis_showline=False,
            legend_borderwidth=0,
            margin=dict(l=20, r=20, t=20, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

    # Row 2 [Group 3]: 作品点赞数分布 + 各Channel行为对比 — 严格对齐
    st.markdown("---")
    g3l, g3r = st.columns(2, gap="medium")
    with g3l:
        st.markdown(
            "<h3 style='font-size:16px;font-weight:700;color:#333;margin:0 0 12px 0;'>"
            "作品点赞数分布</h3>", unsafe_allow_html=True)
        item_likes = (
            filtered_df.groupby("item_id")["like"].sum().reset_index(name="total_likes")
        )
        likes_max = int(item_likes["total_likes"].max())
        likes_bins = st.slider("点赞数范围", 0, max(20, likes_max), 10, 1, key="likes_range")
        likes_clipped = item_likes[item_likes["total_likes"] <= likes_bins]
        fig = px.histogram(
            likes_clipped, x="total_likes",
            nbins=min(likes_bins + 1, 30),
            color_discrete_sequence=[PLOTLY_COLORS[9]],  # orange
            labels={"total_likes": "点赞数", "y": "作品数量"},
        )
        fig.update_layout(
            bargap=0.05, height=CHART_H_FULL,
            plot_bgcolor="white", paper_bgcolor="white",
            xaxis_showline=False, yaxis_showline=False,
            legend_borderwidth=0,
            margin=dict(l=20, r=20, t=20, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

    with g3r:
        st.markdown(
            "<h3 style='font-size:16px;font-weight:700;color:#333;margin:0 0 12px 0;'>"
            "各Channel行为对比</h3>", unsafe_allow_html=True)
        ch_stats = (
            filtered_df.groupby("channel")
            .agg(完成率=("finish", "mean"), 点赞率=("like", "mean"), 记录数=("uid", "count"))
            .reset_index()
        )
        ch_stats["channel_label"] = ch_stats["channel"].apply(lambda x: f"Channel {x}")
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=ch_stats["channel_label"], y=ch_stats["完成率"] * 100,
            name="完成率(%)", marker_color="#6366f1",
            text=ch_stats["完成率"].apply(lambda x: f"{x:.1%}"),
            textposition="outside",
        ))
        fig.add_trace(go.Bar(
            x=ch_stats["channel_label"], y=ch_stats["点赞率"] * 100,
            name="点赞率(%)", marker_color="#8b5cf6",
            text=ch_stats["点赞率"].apply(lambda x: f"{x:.2%}"),
            textposition="outside",
        ))
        fig.update_layout(
            height=CHART_H_FULL, barmode="group",
            plot_bgcolor="white", paper_bgcolor="white",
            xaxis_showline=False, yaxis_showline=False,
            legend_borderwidth=0,
            margin=dict(l=20, r=20, t=20, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

    # --- 最佳视频时长分析 ---
    st.markdown("### ⏱️ 最佳视频时长分析")
    st.caption("按5秒间隔分组，分析各时长区间的平均完播率")

    # 1. 将 duration_time 按5秒间隔分组 (0-5, 5-10, ..., 55-60, >60)
    bin_edges = list(range(0, 61, 5))
    bin_labels = [f"{i}-{i+5}s" for i in bin_edges[:-1]]
    bin_edges.append(float("inf"))
    bin_labels.append(">60s")

    dur_binned = pd.cut(
        filtered_df["duration_time"], bins=bin_edges,
        labels=bin_labels, right=False,
    )
    duration_stats = (
        filtered_df.groupby(dur_binned, observed=False)
        .agg(平均完播率=("finish", "mean"), 记录数=("uid", "count"))
        .reset_index()
    )
    duration_stats.columns = ["时长区间", "平均完播率", "记录数"]
    # 过滤样本过少的区间
    duration_stats = duration_stats[duration_stats["记录数"] >= 50]

    # 2. 柱状图
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=duration_stats["时长区间"], y=duration_stats["平均完播率"] * 100,
        name="平均完播率(%)", marker_color=PLOTLY_COLORS[6],  # pink
        text=duration_stats["平均完播率"].apply(lambda x: f"{x:.2%}"),
        textposition="outside",
    ))

    # 3. LOWESS 平滑 — 使用数值型 x 计算，再映射回分类标签
    duration_stats["x_num"] = range(len(duration_stats))
    fig_trend = px.scatter(
        duration_stats, x="x_num", y=duration_stats["平均完播率"] * 100,
        trendline="lowess", trendline_options=dict(frac=0.35),
        trendline_color_override=PLOTLY_COLORS[4],  # red
    )
    for trace in fig_trend.data:
        if getattr(trace, "mode", "") == "lines":
            trace.x = duration_stats["时长区间"]  # 映射回分类标签
            trace.name = "LOWESS 平滑"
            trace.line.width = 3
            trace.showlegend = True
            fig.add_trace(trace)

    # 4. 标记完播率最高的区间
    best_idx = duration_stats["平均完播率"].idxmax()
    best_bin = duration_stats.loc[best_idx, "时长区间"]
    best_rate = duration_stats.loc[best_idx, "平均完播率"]

    fig.add_annotation(
        x=best_bin, y=best_rate * 100,
        text=f"🏆 最佳: {best_bin}<br>完播率 {best_rate:.2%}",
        showarrow=True, arrowhead=2, arrowsize=1.5,
        arrowcolor="#FF4500",
        font=dict(color="#FF4500", size=13, weight="bold"),
        bgcolor="white", bordercolor="#FF4500",
        borderwidth=2, borderpad=6,
        yshift=25,
    )

    fig.update_layout(
        title="最佳视频时长分析 — 各时长区间平均完播率",
        xaxis_title="视频时长区间",
        yaxis_title="平均完播率 (%)",
        height=CHART_H_FULL, margin=CHART_MARGIN,
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- 时段×星期热力图 ---
    st.markdown("### 时段 × 星期 活跃度热力图")
    col_h1, col_h2 = st.columns([3, 1], gap="small")

    with col_h1:
        weekday_order = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
        hm = heatmap_data.reindex(
            columns=[w for w in weekday_order if w in heatmap_data.columns]
        )
        fig = px.imshow(
            hm.values, x=hm.columns, y=hm.index,
            color_continuous_scale="YlOrRd", aspect="auto",
            title="时段 × 星期 活跃度热力图",
            labels=dict(x="星期", y="小时", color="记录数"),
        )
        fig.update_layout(height=CHART_H_HALF, margin=CHART_MARGIN)
        st.plotly_chart(fig, use_container_width=True)

    with col_h2:
        st.markdown("#### 数据解读")
        st.markdown(f"""
        - **高峰时段**: 20:00-23:00
        - **活跃天数**: {len(daily_stats_f)} 天
        - **日均记录**: {daily_stats_f['total_records'].mean():,.0f}
        - **最活跃日**: {daily_stats_f.loc[daily_stats_f['total_records'].idxmax(), 'date'].strftime('%m/%d')}
        """)


# ---- 4d. 综合仪表盘 ----
def render_dashboard(filtered_df, daily_stats_f, heatmap_data,
                     start_date, end_date, selected_cities, selected_channels):
    """渲染综合仪表盘标签页的所有图表 — 统计指标懒加载+缓存"""
    st.markdown("## 📊 综合数据仪表盘")

    # 懒加载仪表盘专用统计 (带缓存，首次计算后续秒开)
    s = compute_filtered_summary(
        start_date, end_date,
        tuple(selected_cities), tuple(selected_channels),
    )

    # Row 1: 数据摘要 + 每日趋势总览
    row1_col1, row1_col2 = st.columns([1, 2], gap="medium")
    with row1_col1:
        st.markdown("### 📋 数据摘要")
        st.markdown(
            f"""
            <div style="background:#fff;border:1px solid #e2e8f0;padding:22px;
                        border-radius:14px;box-shadow:0 1px 3px rgba(0,0,0,0.05);">
                <h4 style="margin:0 0 14px 0;font-size:0.9rem;font-weight:700;color:#1e293b;">
                数据概览</h4>
                <table style="width:100%;border-collapse:collapse;font-size:0.84rem;color:#64748b;">
                    <tr><td style="padding:7px 0;">📊 总记录数</td><td style="text-align:right;padding:7px 0;font-weight:600;color:#1e293b;">{s['len']:,}</td></tr>
                    <tr><td style="padding:7px 0;">👤 唯一用户</td><td style="text-align:right;padding:7px 0;font-weight:600;color:#1e293b;">{s['n_users']:,}</td></tr>
                    <tr><td style="padding:7px 0;">✍️ 唯一作者</td><td style="text-align:right;padding:7px 0;font-weight:600;color:#1e293b;">{s['n_authors']:,}</td></tr>
                    <tr><td style="padding:7px 0;">🎬 唯一作品</td><td style="text-align:right;padding:7px 0;font-weight:600;color:#1e293b;">{s['n_items']:,}</td></tr>
                    <tr><td style="padding:7px 0;">🏙️ 城市数</td><td style="text-align:right;padding:7px 0;font-weight:600;color:#1e293b;">{s['n_cities']:,}</td></tr>
                    <tr><td style="padding:7px 0;">📅 时间跨度</td><td style="text-align:right;padding:7px 0;font-weight:600;color:#6366f1;">{s['min_date']} ~ {s['max_date']}</td></tr>
                    <tr><td style="padding:7px 0;">✅ 完成率</td><td style="text-align:right;padding:7px 0;font-weight:600;color:#1e293b;">{s['avg_finish']:.2%}</td></tr>
                    <tr><td style="padding:7px 0;">❤️ 点赞率</td><td style="text-align:right;padding:7px 0;font-weight:600;color:#1e293b;">{s['avg_like']:.3%}</td></tr>
                </table>
            </div>""",
            unsafe_allow_html=True,
        )

    with row1_col2:
        st.markdown("### 📈 每日趋势总览")
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(
            x=daily_stats_f["date"], y=daily_stats_f["total_records"],
            name="记录数", fill="tozeroy",
            line=dict(color="#6366f1", width=2),
        ), secondary_y=False)
        fig.add_trace(go.Scatter(
            x=daily_stats_f["date"], y=daily_stats_f["avg_finish"] * 100,
            name="完成率(%)", line=dict(color="#10b981", width=1.5, dash="dot"),
        ), secondary_y=True)
        fig.add_trace(go.Scatter(
            x=daily_stats_f["date"], y=daily_stats_f["avg_like"] * 100,
            name="点赞率(%)", line=dict(color="#ef4444", width=1.5, dash="dot"),
        ), secondary_y=True)
        fig.update_layout(
            title="每日记录数 & 行为率趋势", hovermode="x unified",
            height=CHART_H_FULL, legend=dict(orientation="h", y=1.1),
            plot_bgcolor="white", paper_bgcolor="white",
            xaxis_showline=False, yaxis_showline=False,
            legend_borderwidth=0,
            margin=dict(l=20, r=20, t=20, b=20),
        )
        fig.update_yaxes(title_text="记录数", secondary_y=False)
        fig.update_yaxes(title_text="比率 (%)", secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)

    # Row 2 [Group 4]: 用户行为指标 + Top10 城市 + Channel 分布 — st.columns(3)
    st.markdown("---")
    g4a, g4b, g4c = st.columns(3, gap="medium")

    wkend_finish = f"{s['weekend_finish']:.2%}" if not np.isnan(s['weekend_finish']) else "N/A"
    wkday_finish = f"{s['weekday_finish']:.2%}" if not np.isnan(s['weekday_finish']) else "N/A"

    with g4a:
        st.markdown(
            "<h3 style='font-size:16px;font-weight:700;color:#333;margin:0 0 12px 0;'>"
            "用户行为指标</h3>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <div style="background:#fff;padding:12px 0;">
            <table style="width:100%;border-collapse:collapse;font-size:0.84rem;color:#64748b;">
                <tr><td style="padding:7px 0;">人均观看次数</td><td style="text-align:right;padding:7px 0;font-weight:600;color:#1e293b;">{s['avg_watch_per_user']:.1f}</td></tr>
                <tr><td style="padding:7px 0;">人均观看不同作品</td><td style="text-align:right;padding:7px 0;font-weight:600;color:#1e293b;">{s['avg_items_per_user']:.1f}</td></tr>
                <tr><td style="padding:7px 0;">平均视频时长</td><td style="text-align:right;padding:7px 0;font-weight:600;color:#1e293b;">{s['avg_duration']:.1f}s</td></tr>
                <tr><td style="padding:7px 0;">作者人均作品</td><td style="text-align:right;padding:7px 0;font-weight:600;color:#1e293b;">{s['avg_items_per_author']:.1f}</td></tr>
                <tr><td style="padding:7px 0;">周末完成率</td><td style="text-align:right;padding:7px 0;font-weight:600;color:#1e293b;">{wkend_finish}</td></tr>
                <tr><td style="padding:7px 0;">工作日完成率</td><td style="text-align:right;padding:7px 0;font-weight:600;color:#1e293b;">{wkday_finish}</td></tr>
            </table></div>""",
            unsafe_allow_html=True,
        )

    with g4b:
        st.markdown(
            "<h3 style='font-size:16px;font-weight:700;color:#333;margin:0 0 12px 0;'>"
            "Top10 城市</h3>", unsafe_allow_html=True)
        top10 = s["top10_cities"]
        fig = px.bar(
            top10, x="count", y="city", orientation="h",
            color="count", color_continuous_scale=["#c7d2fe", "#6366f1"],
        )
        fig.update_layout(
            height=CHART_H_FULL, yaxis=dict(categoryorder="total ascending"),
            plot_bgcolor="white", paper_bgcolor="white",
            xaxis_showline=False, yaxis_showline=False,
            legend_borderwidth=0,
            margin=dict(l=20, r=20, t=20, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

    with g4c:
        st.markdown(
            "<h3 style='font-size:16px;font-weight:700;color:#333;margin:0 0 12px 0;'>"
            "Channel 分布</h3>", unsafe_allow_html=True)
        ch_items = s["channel_dist"]
        fig = px.pie(
            ch_items, values="count", names="channel",
            color_discrete_sequence=PLOTLY_COLORS,
        )
        fig.update_layout(
            height=CHART_H_FULL,
            plot_bgcolor="white", paper_bgcolor="white",
            xaxis_showline=False, yaxis_showline=False,
            legend_borderwidth=0,
            margin=dict(l=20, r=20, t=20, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

    # 底部: 时段分布 + Violin
    st.markdown("---")
    bottom_col1, bottom_col2 = st.columns(2, gap="small")

    with bottom_col1:
        st.markdown("### ⏰ 24小时时段分布")
        hourly_dash = s["hourly_dash"]
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Bar(
                x=hourly_dash["H"], y=hourly_dash["记录数"],
                name="记录数", marker_color="#6366f1", opacity=0.75,
            ),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=hourly_dash["H"], y=hourly_dash["完成率"] * 100,
                name="完成率(%)", mode="lines+markers",
                line=dict(color="#10b981", width=2),
            ),
            secondary_y=True,
        )
        fig.update_layout(
            title="24小时分布", height=CHART_H_HALF, hovermode="x unified",
            margin=CHART_MARGIN, xaxis=dict(dtick=2),
        )
        fig.update_yaxes(title_text="记录数", secondary_y=False)
        fig.update_yaxes(title_text="比率(%)", secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)

    with bottom_col2:
        st.markdown("### 📉 时长分布 (按Channel)")
        # 从 filtered_df 实时取 Top4 Channel 的时长数据 (filtered_df 已缓存)
        top4 = (
            filtered_df["channel"].value_counts().nlargest(4).index.tolist()
        )
        fig = go.Figure()
        violin_colors = ["#6366f1", "#8b5cf6", "#10b981", "#f59e0b"]
        for i, ch in enumerate(top4):
            ch_data = filtered_df.loc[
                filtered_df["channel"] == ch, "duration_time"
            ].clip(0, 30)
            fig.add_trace(go.Violin(
                y=ch_data, name=f"Channel {ch}",
                box_visible=True, meanline_visible=True,
                line_color=violin_colors[i],
                fillcolor=violin_colors[i],
                opacity=0.55,
            ))
        fig.update_layout(
            title="各Channel视频时长分布 (≤30s)",
            height=CHART_H_HALF, margin=CHART_MARGIN,
            yaxis_title="时长 (秒)",
        )
        st.plotly_chart(fig, use_container_width=True)


# ============================================================
# 5. 主入口
# ============================================================
def main():
    # ---- 5a. 加载数据 (带状态提示) ----
    with st.status("🚀 正在加载数据...", expanded=False) as load_status:
        t_start = time.time()
        df = load_data()
        load_status.update(
            label=f"✅ 数据加载完成 ({time.time() - t_start:.1f}s, {len(df):,} 条记录)",
            state="complete",
        )

    # ---- 5b. 全量预计算所有聚合 (一次计算，永久缓存) ----
    with st.status("⚡ 正在预计算分析数据...", expanded=False) as precompute_status:
        t0 = time.time()
        daily_stats_all = compute_daily_stats(df)
        t1 = time.time()
        hourly_stats_all = compute_hourly_stats(df)
        t2 = time.time()
        city_stats_all = compute_city_stats(df)
        t3 = time.time()
        channel_stats_all = compute_channel_stats(df)
        t4 = time.time()
        user_stats_all = compute_user_stats(df)
        t5 = time.time()
        author_stats_all = compute_author_stats(df)
        t6 = time.time()
        item_stats_all = compute_item_stats(df)
        t7 = time.time()
        heatmap_data_all = compute_hourly_weekday_heatmap(df)
        t8 = time.time()

        precompute_status.update(
            label=f"✅ 预计算完成 (总耗时 {t8 - t0:.1f}s, 各聚合均已缓存)",
            state="complete",
        )

    st.session_state.data_loaded = True

    # ---- 5c. 侧边栏筛选器 ----
    st.sidebar.markdown(
        "<div style='font-size: 1.2rem; font-weight: 800; "
        "background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%); "
        "-webkit-background-clip: text; -webkit-text-fill-color: transparent; "
        "background-clip: text; padding-bottom: 0.25rem;'>"
        "🎵 抖音数据分析</div>",
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("---")

    # 日期范围
    st.sidebar.markdown("### 📅 日期范围")
    min_date = df["date"].min().date()
    max_date = df["date"].max().date()
    date_range = st.sidebar.date_input(
        "选择分析时间段",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )
    # 处理单个日期 vs 日期范围
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date = end_date = date_range

    # 城市筛选
    st.sidebar.markdown("### 🏙️ 城市筛选")
    city_select_mode = st.sidebar.radio(
        "选择模式", ["全部城市", "Top N 城市", "自定义城市"],
        horizontal=True, key="city_mode",
    )

    all_cities_list = sorted(df["user_city"].unique().tolist())

    if city_select_mode == "Top N 城市":
        top_n = st.sidebar.slider("选择Top城市数量", 5, 50, 20, 5, key="top_n")
        # 使用已缓存的 city_stats_all 代替 df["user_city"].value_counts() 避免全表扫描
        selected_cities = city_stats_all.head(top_n)["user_city"].tolist()
    elif city_select_mode == "自定义城市":
        selected_cities = st.sidebar.multiselect(
            "选择城市", all_cities_list, default=all_cities_list[:5],
            key="custom_cities",
        )
    else:
        selected_cities = all_cities_list

    # Channel 筛选
    st.sidebar.markdown("### 📺 Channel")
    all_channels = sorted(df["channel"].unique().tolist())
    selected_channels = st.sidebar.multiselect(
        "选择频道", all_channels, default=all_channels, key="channel_select",
    )

    st.sidebar.markdown("---")

    # 验证筛选结果
    if not selected_cities or not selected_channels:
        st.error("⚠️ 请至少选择一个城市和一个Channel。")
        st.stop()

    # ---- 5d. 在预计算聚合上筛选 ----
    daily_stats_f = filter_daily_stats(daily_stats_all, (start_date, end_date))
    city_stats_f = filter_city_stats(city_stats_all, selected_cities)
    channel_stats_f = filter_channel_stats(channel_stats_all, selected_channels)
    user_stats_f = filter_user_stats(user_stats_all, selected_cities)

    # 作者 & 作品统计不依赖日期/城市筛选 (全量作者画像)
    author_stats_f = author_stats_all
    item_stats_f = item_stats_all

    # 热力图基于全量 (无日期列, 全量即可)
    heatmap_data_f = heatmap_data_all

    # 对需要原始数据的图表，构造轻量筛选 df (带缓存)
    filtered_df = filter_raw_df(
        start_date, end_date,
        tuple(selected_cities), tuple(selected_channels),
    )

    # 轻量摘要 (侧边栏 + KPI 都用这些基本指标，秒算)
    n_total = len(filtered_df)
    n_users_kpi = max(filtered_df["uid"].nunique(), 1)
    n_authors_kpi = filtered_df["author_id"].nunique()
    n_items_kpi = filtered_df["item_id"].nunique()
    avg_finish_kpi = filtered_df["finish"].mean()
    avg_like_kpi = filtered_df["like"].mean()
    avg_duration_kpi = filtered_df["duration_time"].mean()
    n_days_f = max((end_date - start_date).days, 1)

    # 数据概览指标
    st.sidebar.markdown(f"### 📊 当前筛选概览")
    st.sidebar.metric("总记录数", f"{n_total:,}")
    st.sidebar.metric("唯一用户", f"{n_users_kpi:,}")
    st.sidebar.metric("唯一作者", f"{n_authors_kpi:,}")
    st.sidebar.metric("唯一作品", f"{n_items_kpi:,}")

    # ---- DeepSeek AI 分析助手 ----
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🤖 AI 分析助手")

    # 初始化聊天历史
    if "ai_chat_history" not in st.session_state:
        st.session_state.ai_chat_history = []

    user_question = st.sidebar.text_area(
        "用自然语言提问",
        placeholder="例如：哪个城市的完成率最高？\n周末和工作日哪个点赞率更高？\n作者发布作品数的分布是怎样的？",
        key="ai_question",
        height=80,
    )

    if st.sidebar.button("🔍 询问 AI", use_container_width=True, key="ai_ask_btn"):
        if user_question.strip():
            with st.sidebar:
                with st.spinner("🤔 AI 分析中..."):

                    # 构建当前数据上下文 — 使用缓存的统计数据
                    top5_cities = (
                        filtered_df["user_city"].value_counts().head(5).to_dict()
                    )
                    ch_stats_context = (
                        filtered_df.groupby("channel")
                        .agg(完成率=("finish", "mean"), 点赞率=("like", "mean"), 记录数=("uid", "count"))
                        .to_dict()
                    )

                    context = f"""
你是一个抖音数据分析助手。当前用户筛选了以下数据：

【数据概览】
- 总记录数: {n_total:,}
- 唯一用户数: {n_users_kpi:,}
- 唯一作者数: {n_authors_kpi:,}
- 唯一作品数: {n_items_kpi:,}
- 日期范围: {start_date} ~ {end_date}
- 城市数量: {filtered_df['user_city'].nunique()}
- Channel 数量: {filtered_df['channel'].nunique()}

【整体行为指标】
- 平均完成率: {avg_finish_kpi:.2%}
- 平均点赞率: {avg_like_kpi:.3%}
- 平均视频时长: {avg_duration_kpi:.1f}秒
- 中位数时长: {filtered_df['duration_time'].median():.0f}秒
- 人均观看次数: {n_total/n_users_kpi:.1f}

【Top5 城市记录数】
{top5_cities}

【工作日 vs 周末】
- 工作日记录数: {(filtered_df['is_weekend']==0).sum():,}
- 周末记录数: {(filtered_df['is_weekend']==1).sum():,}
- 工作日平均完成率: {filtered_df.loc[filtered_df['is_weekend']==0, 'finish'].mean():.2%}
- 周末平均完成率: {filtered_df.loc[filtered_df['is_weekend']==1, 'finish'].mean():.2%}

【24小时活跃度】
- 最活跃时段 (H): {filtered_df.groupby('H').size().idxmax()}点 ({filtered_df.groupby('H').size().max():,}条)
- 凌晨(0-6)完成率: {filtered_df.loc[filtered_df['H'].between(0,6), 'finish'].mean():.2%}
- 黄金时段(18-23)完成率: {filtered_df.loc[filtered_df['H'].between(18,23), 'finish'].mean():.2%}

请根据以上数据简洁回答用户问题，用中文，控制在200字以内。如果数据不足以回答，请说明需要哪些额外筛选。
"""
                    try:
                        resp = requests.post(
                            "https://api.deepseek.com/chat/completions",
                            headers={
                                "Authorization": "Bearer sk-2dea7bf457e8457f985381108e5724d4",
                                "Content-Type": "application/json",
                            },
                            json={
                                "model": "deepseek-chat",
                                "messages": [
                                    {"role": "system", "content": context},
                                    {"role": "user", "content": user_question},
                                ],
                                "temperature": 0.3,
                                "max_tokens": 400,
                            },
                            timeout=20,
                        )

                        if resp.status_code == 200:
                            answer = resp.json()["choices"][0]["message"]["content"]
                            st.session_state.ai_chat_history.append(
                                {"q": user_question, "a": answer}
                            )
                            st.success(answer)
                        else:
                            st.error(f"API 调用失败: {resp.status_code} - {resp.text[:200]}")
                    except Exception as e:
                        st.error(f"请求异常: {e}")
        else:
            st.sidebar.warning("请输入问题")

    # 显示历史问答
    if st.session_state.ai_chat_history:
        with st.sidebar.expander("📝 历史问答", expanded=False):
            for i, chat in enumerate(reversed(st.session_state.ai_chat_history[-5:])):
                st.markdown(f"**Q:** {chat['q']}")
                st.markdown(f"**A:** {chat['a']}")
                if i < len(st.session_state.ai_chat_history[-5:]) - 1:
                    st.divider()

    # ---- 5e. 主页面 ----
    st.markdown(
        f"<div style='margin-bottom: 1rem;'>"
        f"<h1 style='font-size: 1.4rem; font-weight: 700; color: #1e293b; "
        f"margin: 0 0 0.15rem 0; letter-spacing: -0.01em;'>"
        f"🎵 抖音用户行为数据分析平台</h1>"
        f"<p style='font-size: 0.78rem; color: #94a3b8; margin: 0;'>"
        f"当前筛选 · {n_total:,} 条记录 · {n_users_kpi:,} 用户 · "
        f"{n_authors_kpi:,} 作者 · "
        f"{date_range[0] if isinstance(date_range, tuple) else date_range} ~ "
        f"{date_range[1] if isinstance(date_range, tuple) else date_range}</p>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # ---- 5f. KPI 指标卡 ----
    k1, k2, k3, k4, k5, k6 = st.columns(6, gap="small")
    with k1:
        st.metric("📈 平均完成率", f"{avg_finish_kpi:.1%}")
    with k2:
        st.metric("❤️ 平均点赞率", f"{avg_like_kpi:.2%}")
    with k3:
        st.metric("⏱️ 平均时长(s)", f"{avg_duration_kpi:.1f}")
    with k4:
        st.metric("👤 人均观看", f"{n_total/n_users_kpi:.1f}")
    with k5:
        st.metric("🎬 日均记录", f"{n_total/n_days_f:,.0f}")
    with k6:
        st.metric("📹 作品数", f"{n_items_kpi:,}")

    # ---- 5g. 标签页选择器 (惰性渲染) ----
    st.markdown("---")

    # 使用 radio + session_state 实现惰性渲染
    active_tab = st.radio(
        "📂 选择分析维度",
        DEFAULT_TABS,
        horizontal=True,
        key="tab_selector",
    )
    st.session_state.active_tab = active_tab

    st.markdown("---")

    # ---- 5h. 仅渲染当前激活的标签页 ----
    if st.session_state.active_tab == "👤 用户分析":
        render_user_analysis(
            filtered_df, user_stats_f, city_stats_f,
            hourly_stats_all, heatmap_data_f,
        )

    elif st.session_state.active_tab == "✍️ 作者分析":
        render_author_analysis(author_stats_f, filtered_df)

    elif st.session_state.active_tab == "🎬 作品分析":
        render_item_analysis(filtered_df, daily_stats_f, heatmap_data_f)

    elif st.session_state.active_tab == "📊 综合仪表盘":
        render_dashboard(filtered_df, daily_stats_f, heatmap_data_f,
                         start_date, end_date, selected_cities, selected_channels)

    # ---- 5i. 页脚 ----
    st.markdown(
        "<div style='height: 1px; background: #e5e7eb; margin: 2rem 0 1rem 0;'></div>"
        "<div style='text-align: center; font-size: 0.75rem; color: #d1d5db;'>"
        "抖音用户行为数据分析平台 · Streamlit + Plotly"
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
