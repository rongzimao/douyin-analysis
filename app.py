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


@st.cache_data(ttl=3600, max_entries=1)
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
@st.cache_data(ttl=3600, max_entries=3)
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


@st.cache_data(ttl=3600, max_entries=3)
def compute_hourly_stats(df: pd.DataFrame) -> pd.DataFrame:
    """小时聚合统计 (全量)"""
    hourly = df.groupby("H").agg(
        record_count=("uid", "count"),
        unique_users=("uid", "nunique"),
        avg_finish=("finish", "mean"),
        avg_like=("like", "mean"),
    ).reset_index()
    return hourly


@st.cache_data(ttl=3600, max_entries=3)
def compute_city_stats(df: pd.DataFrame) -> pd.DataFrame:
    """城市聚合统计 (全量)"""
    return df.groupby("user_city").agg(
        record_count=("uid", "count"),
        unique_users=("uid", "nunique"),
        avg_finish=("finish", "mean"),
        avg_like=("like", "mean"),
    ).reset_index().sort_values("record_count", ascending=False)


@st.cache_data(ttl=3600, max_entries=3)
def compute_channel_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Channel 聚合统计 (全量)"""
    return df.groupby("channel").agg(
        record_count=("uid", "count"),
        avg_finish=("finish", "mean"),
        avg_like=("like", "mean"),
    ).reset_index()


@st.cache_data(ttl=3600, max_entries=3)
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


@st.cache_data(ttl=3600, max_entries=3)
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


@st.cache_data(ttl=3600, max_entries=3)
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


@st.cache_data(ttl=3600, max_entries=3)
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


def filter_raw_df(
    df: pd.DataFrame, date_range, selected_cities: list, selected_channels: list
) -> pd.DataFrame:
    """筛选原始数据 — 仅用于需要原始数据的少数图表 (如时长分布)"""
    mask = (
        (df["date"].dt.date >= date_range[0])
        & (df["date"].dt.date <= date_range[1])
        & (df["user_city"].isin(selected_cities))
        & (df["channel"].isin(selected_channels))
    )
    return df[mask]


# ============================================================
# 4. 图表构建函数 — 按标签页分组，仅在需要时调用
# ============================================================

# ---- 4a. 用户分析图表 ----
def render_user_analysis(
    filtered_df, user_stats_f, city_stats_f, hourly_stats, heatmap_data,
):
    """渲染用户分析标签页的所有图表"""
    st.markdown("## 👤 用户维度分析")

    col_left, col_right = st.columns(2)

    with col_left:
        # --- 用户观看次数分布 ---
        st.markdown("### 用户观看次数分布")
        bin_option = st.radio(
            "X轴刻度", ["线性", "对数"], key="user_watch_scale", horizontal=True
        )
        log_x = bin_option == "对数"

        fig = px.histogram(
            user_stats_f, x="watch_count", nbins=50,
            log_x=log_x, log_y=True,
            color_discrete_sequence=["#3366CC"],
            title="用户观看次数分布 (Y轴对数)",
            labels={"watch_count": "观看次数"},
        )
        fig.update_layout(bargap=0.05, height=400, margin=dict(t=40, b=10))
        median_val = user_stats_f["watch_count"].median()
        fig.add_vline(
            x=median_val, line_dash="dash", line_color="red",
            annotation_text=f"中位数: {median_val:.0f}",
        )
        st.plotly_chart(fig, use_container_width=True)

        # --- 用户城市分布 ---
        st.markdown("### 用户城市分布 Top20")
        top20_cities = city_stats_f.head(20).copy()
        top20_cities["city_label"] = top20_cities["user_city"].astype(str)

        fig = px.bar(
            top20_cities, x="record_count", y="city_label", orientation="h",
            color="record_count", color_continuous_scale="Blues",
            title="用户城市分布 Top20",
            labels={"record_count": "记录数", "city_label": "城市ID"},
        )
        fig.update_layout(
            yaxis=dict(categoryorder="total ascending"),
            height=500, margin=dict(t=40, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        # --- 用户活跃度分层 ---
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
            color_discrete_sequence=px.colors.qualitative.Set2,
            title="用户活跃度分层 (按观看次数)",
        )
        fig.update_traces(textposition="inside", textinfo="percent+value")
        fig.update_layout(height=400, margin=dict(t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)

        # --- 用户完成率 vs 点赞率 (scattergl, 全量数据) ---
        st.markdown("### 用户完成率 vs 点赞率")
        fig = px.scatter(
            user_stats_f, x="finish_rate", y="like_rate",
            size=np.log1p(user_stats_f["watch_count"]), size_max=15,
            color="watch_count", color_continuous_scale="Viridis",
            opacity=0.5,
            title=f"用户完成率 vs 点赞率 ({len(user_stats_f):,} 用户, 全部数据)",
            labels={
                "finish_rate": "完成率", "like_rate": "点赞率",
                "watch_count": "观看次数",
            },
            render_mode="webgl",
        )
        fig.update_layout(height=450, margin=dict(t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)

    # --- 24小时时段偏好 ---
    st.markdown("### 用户观看时段偏好 (24小时)")
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(
            x=hourly_stats["H"], y=hourly_stats["record_count"],
            name="记录数", marker_color="steelblue", opacity=0.8,
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=hourly_stats["H"], y=hourly_stats["avg_finish"] * 100,
            name="完成率(%)", mode="lines+markers",
            line=dict(color="green", width=2),
        ),
        secondary_y=True,
    )
    fig.add_trace(
        go.Scatter(
            x=hourly_stats["H"], y=hourly_stats["avg_like"] * 100,
            name="点赞率(%)", mode="lines+markers",
            line=dict(color="red", width=2),
        ),
        secondary_y=True,
    )
    fig.update_layout(
        title="24小时活跃度与行为趋势",
        xaxis=dict(title="小时", dtick=2),
        height=400, margin=dict(t=40, b=10),
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="记录数", secondary_y=False)
    fig.update_yaxes(title_text="比率 (%)", secondary_y=True)
    st.plotly_chart(fig, use_container_width=True)


# ---- 4b. 作者分析图表 ----
def render_author_analysis(author_stats_f, filtered_df):
    """渲染作者分析标签页的所有图表"""
    st.markdown("## ✍️ 作者维度分析")

    auth_left, auth_right = st.columns(2)

    with auth_left:
        # --- 作者作品数分布 ---
        st.markdown("### 作者发布作品数分布")
        log_author_x = st.checkbox("X轴对数", value=True, key="author_log")

        fig = px.histogram(
            author_stats_f, x="video_count", nbins=50,
            log_x=log_author_x, log_y=True,
            color_discrete_sequence=["#8A2BE2"],
            title="作者发布作品数分布",
            labels={"video_count": "作品数"},
        )
        fig.update_layout(bargap=0.05, height=400, margin=dict(t=40, b=10))
        fig.add_vline(
            x=author_stats_f["video_count"].median(), line_dash="dash",
            line_color="red",
            annotation_text=f"中位数: {author_stats_f['video_count'].median():.0f}",
        )
        st.plotly_chart(fig, use_container_width=True)

        # --- 最活跃作者 ---
        st.markdown("### 最活跃作者 Top15")
        top_auth = author_stats_f.nlargest(15, "video_count")
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=[f"A-{int(a)}" for a in top_auth["author_id"]],
            x=top_auth["video_count"], orientation="h",
            marker=dict(
                color=top_auth["video_count"],
                colorscale="Purples", showscale=True,
            ),
            name="作品数",
        ))
        fig.update_layout(
            title="最活跃作者 Top15 (按去重作品数)",
            height=450, margin=dict(t=40, b=10),
            xaxis_title="作品数",
            yaxis=dict(categoryorder="total ascending"),
        )
        st.plotly_chart(fig, use_container_width=True)

    with auth_right:
        # --- 作者平均作品时长分布 ---
        st.markdown("### 作者平均作品时长分布")
        duration_max = st.slider(
            "时长上限(秒)", 10, 120, 60, 10, key="author_dur_max"
        )
        dur_clipped = author_stats_f[
            author_stats_f["avg_duration"] <= duration_max
        ]

        fig = px.histogram(
            dur_clipped, x="avg_duration", nbins=50,
            color_discrete_sequence=["#008B8B"],
            title=f"作者平均作品时长分布 (≤{duration_max}s)",
            labels={"avg_duration": "平均时长 (秒)"},
        )
        fig.update_layout(bargap=0.05, height=400, margin=dict(t=40, b=10))
        fig.add_vline(
            x=author_stats_f["avg_duration"].median(), line_dash="dash",
            line_color="red",
            annotation_text=f"中位数: {author_stats_f['avg_duration'].median():.1f}s",
        )
        st.plotly_chart(fig, use_container_width=True)

        # --- 最受欢迎作者 ---
        st.markdown("### 最受欢迎作者 Top15 (按点赞数)")
        top_likes = author_stats_f.nlargest(15, "total_likes")
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=[f"A-{int(a)}" for a in top_likes["author_id"]],
            x=top_likes["total_likes"], orientation="h",
            marker=dict(
                color=top_likes["total_likes"],
                colorscale="Reds", showscale=True,
            ),
            name="总点赞数",
            hovertemplate=(
                "作者: %{y}<br>点赞: %{x}<br>"
                "观看: %{customdata[0]:,}<br>作品: %{customdata[1]}"
            ),
            customdata=top_likes[["total_views", "video_count"]],
        ))
        fig.update_layout(
            title="最受欢迎作者 Top15 (按总点赞数)",
            height=450, margin=dict(t=40, b=10),
            xaxis_title="总点赞数",
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
    fig.update_layout(height=450, margin=dict(t=40, b=10))
    st.plotly_chart(fig, use_container_width=True)


# ---- 4c. 作品分析图表 ----
def render_item_analysis(filtered_df, daily_stats_f, heatmap_data):
    """渲染作品分析标签页的所有图表"""
    st.markdown("## 🎬 作品维度分析")

    item_left, item_right = st.columns(2)

    with item_left:
        # --- 视频时长分布 ---
        st.markdown("### 视频时长分布")
        dur_upper = st.slider("时长上限(秒)", 10, 300, 60, 10, key="item_dur_slider")
        dur_data = filtered_df["duration_time"].clip(0, dur_upper)

        fig = px.histogram(
            x=dur_data, nbins=min(dur_upper, 60),
            color_discrete_sequence=["#4682B4"],
            title=f"视频时长分布 (≤{dur_upper}s)",
            labels={"x": "时长 (秒)", "y": "记录数"},
        )
        fig.update_layout(bargap=0.05, height=400, margin=dict(t=40, b=10))
        fig.add_vline(
            x=filtered_df["duration_time"].median(), line_dash="dash",
            line_color="red",
            annotation_text=f"中位数: {filtered_df['duration_time'].median():.0f}s",
        )
        fig.add_vline(
            x=filtered_df["duration_time"].mean(), line_dash="dot",
            line_color="orange",
            annotation_text=f"均值: {filtered_df['duration_time'].mean():.1f}s",
        )
        st.plotly_chart(fig, use_container_width=True)

        # --- 作品点赞分布 ---
        st.markdown("### 作品点赞数分布")
        item_likes = (
            filtered_df.groupby("item_id")["like"].sum().reset_index(name="total_likes")
        )
        likes_max = int(item_likes["total_likes"].max())
        likes_bins = st.slider(
            "点赞数范围", 0, max(20, likes_max), 10, 1, key="likes_range"
        )
        likes_clipped = item_likes[item_likes["total_likes"] <= likes_bins]

        fig = px.histogram(
            likes_clipped, x="total_likes",
            nbins=min(likes_bins + 1, 30),
            color_discrete_sequence=["#FF6347"],
            title=f"作品点赞数分布 (0-{likes_bins}赞)",
            labels={"total_likes": "点赞数", "y": "作品数量"},
        )
        fig.update_layout(bargap=0.05, height=400, margin=dict(t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with item_right:
        # --- 每日趋势 ---
        st.markdown("### 每日作品/记录数趋势")
        trend_metric = st.radio(
            "显示指标", ["总记录数", "去重作品数", "两者"],
            horizontal=True, key="trend_metric",
        )
        fig = go.Figure()
        if trend_metric in ["总记录数", "两者"]:
            fig.add_trace(go.Scatter(
                x=daily_stats_f["date"], y=daily_stats_f["total_records"],
                mode="lines", name="总记录数",
                line=dict(color="steelblue", width=2),
                fill="tozeroy", fillcolor="rgba(70,130,180,0.2)",
            ))
        if trend_metric in ["去重作品数", "两者"]:
            fig.add_trace(go.Scatter(
                x=daily_stats_f["date"], y=daily_stats_f["unique_items"],
                mode="lines", name="去重作品数",
                line=dict(color="coral", width=2),
                fill="tozeroy", fillcolor="rgba(255,127,80,0.15)",
            ))
        fig.update_layout(
            title="每日趋势", xaxis_title="日期", yaxis_title="数量",
            height=400, margin=dict(t=40, b=10), hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)

        # --- Channel 对比 ---
        st.markdown("### 各Channel行为对比")
        ch_stats = (
            filtered_df.groupby("channel")
            .agg(完成率=("finish", "mean"), 点赞率=("like", "mean"), 记录数=("uid", "count"))
            .reset_index()
        )
        ch_stats["channel_label"] = ch_stats["channel"].apply(lambda x: f"Channel {x}")

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=ch_stats["channel_label"], y=ch_stats["完成率"] * 100,
            name="完成率(%)", marker_color="seagreen",
            text=ch_stats["完成率"].apply(lambda x: f"{x:.1%}"),
            textposition="outside",
        ))
        fig.add_trace(go.Bar(
            x=ch_stats["channel_label"], y=ch_stats["点赞率"] * 100,
            name="点赞率(%)", marker_color="tomato",
            text=ch_stats["点赞率"].apply(lambda x: f"{x:.1%}"),
            textposition="outside",
        ))
        fig.update_layout(
            title="各Channel完成率与点赞率",
            height=400, margin=dict(t=40, b=10), barmode="group",
        )
        st.plotly_chart(fig, use_container_width=True)

    # --- 时段×星期热力图 ---
    st.markdown("### 时段 × 星期 活跃度热力图")
    col_h1, col_h2 = st.columns([3, 1])

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
        fig.update_layout(height=400, margin=dict(t=40, b=10))
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
def render_dashboard(filtered_df, daily_stats_f, heatmap_data):
    """渲染综合仪表盘标签页的所有图表"""
    st.markdown("## 📊 综合数据仪表盘")

    # 第一行: 摘要卡片 + 每日趋势
    row1_col1, row1_col2 = st.columns([1, 2])

    with row1_col1:
        st.markdown("### 📋 数据摘要")
        st.markdown(
            f"""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        padding: 25px; border-radius: 15px; color: white;">
                <h3 style="margin-top: 0;">数据概览</h3>
                <table style="width:100%; color: white;">
                    <tr><td>📊 总记录数</td><td style="text-align:right;"><b>{len(filtered_df):,}</b></td></tr>
                    <tr><td>👤 唯一用户</td><td style="text-align:right;"><b>{filtered_df['uid'].nunique():,}</b></td></tr>
                    <tr><td>✍️ 唯一作者</td><td style="text-align:right;"><b>{filtered_df['author_id'].nunique():,}</b></td></tr>
                    <tr><td>🎬 唯一作品</td><td style="text-align:right;"><b>{filtered_df['item_id'].nunique():,}</b></td></tr>
                    <tr><td>🏙️ 城市数</td><td style="text-align:right;"><b>{filtered_df['user_city'].nunique():,}</b></td></tr>
                    <tr><td>📅 时间跨度</td><td style="text-align:right;"><b>{filtered_df['date'].min().date()} ~ {filtered_df['date'].max().date()}</b></td></tr>
                    <tr><td>✅ 完成率</td><td style="text-align:right;"><b>{filtered_df['finish'].mean():.2%}</b></td></tr>
                    <tr><td>❤️ 点赞率</td><td style="text-align:right;"><b>{filtered_df['like'].mean():.2%}</b></td></tr>
                </table>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("### 🎯 用户行为指标")
        n_users = max(filtered_df["uid"].nunique(), 1)
        weekend_mask = filtered_df["is_weekend"] == 1
        st.markdown(
            f"""
            <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                        padding: 25px; border-radius: 15px; color: white; margin-top: 15px;">
                <table style="width:100%; color: white;">
                    <tr><td>人均观看次数</td><td style="text-align:right;"><b>{len(filtered_df)/n_users:.1f}</b></td></tr>
                    <tr><td>人均观看不同作品</td><td style="text-align:right;"><b>{filtered_df.groupby('uid')['item_id'].nunique().mean():.1f}</b></td></tr>
                    <tr><td>平均视频时长</td><td style="text-align:right;"><b>{filtered_df['duration_time'].mean():.1f}s</b></td></tr>
                    <tr><td>作者人均作品</td><td style="text-align:right;"><b>{filtered_df.groupby('author_id')['item_id'].nunique().mean():.1f}</b></td></tr>
                    <tr><td>周末完成率</td><td style="text-align:right;"><b>{filtered_df[weekend_mask]['finish'].mean() if weekend_mask.any() else 'N/A'}</b></td></tr>
                    <tr><td>工作日完成率</td><td style="text-align:right;"><b>{filtered_df[~weekend_mask]['finish'].mean() if (~weekend_mask).any() else 'N/A'}</b></td></tr>
                </table>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with row1_col2:
        # 每日趋势总览
        st.markdown("### 📈 每日趋势总览")
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Scatter(
                x=daily_stats_f["date"], y=daily_stats_f["total_records"],
                name="记录数", fill="tozeroy",
                line=dict(color="steelblue", width=2),
            ),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=daily_stats_f["date"], y=daily_stats_f["avg_finish"] * 100,
                name="完成率(%)", line=dict(color="green", width=1.5, dash="dot"),
            ),
            secondary_y=True,
        )
        fig.add_trace(
            go.Scatter(
                x=daily_stats_f["date"], y=daily_stats_f["avg_like"] * 100,
                name="点赞率(%)", line=dict(color="red", width=1.5, dash="dot"),
            ),
            secondary_y=True,
        )
        fig.update_layout(
            title="每日记录数 & 行为率趋势", hovermode="x unified",
            height=350, margin=dict(t=40, b=10),
            legend=dict(orientation="h", y=1.1),
        )
        fig.update_yaxes(title_text="记录数", secondary_y=False)
        fig.update_yaxes(title_text="比率 (%)", secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)

        # 城市 + Channel 分布
        dash_col1, dash_col2 = st.columns(2)
        with dash_col1:
            top10 = (
                filtered_df["user_city"].value_counts().nlargest(10).reset_index()
            )
            top10.columns = ["city", "count"]
            top10["city"] = top10["city"].astype(str)
            fig = px.bar(
                top10, x="count", y="city", orientation="h",
                color="count", color_continuous_scale="Blues",
                title="Top10 城市",
            )
            fig.update_layout(
                height=300, margin=dict(t=40, b=10),
                yaxis=dict(categoryorder="total ascending"),
            )
            st.plotly_chart(fig, use_container_width=True)

        with dash_col2:
            ch_items = filtered_df.groupby("channel").size().reset_index(name="count")
            ch_items["channel"] = ch_items["channel"].apply(lambda x: f"Ch-{x}")
            fig = px.pie(
                ch_items, values="count", names="channel",
                color_discrete_sequence=px.colors.qualitative.Pastel,
                title="Channel 分布",
            )
            fig.update_layout(height=300, margin=dict(t=40, b=10))
            st.plotly_chart(fig, use_container_width=True)

    # 底部: 时段分布 + Violin
    st.markdown("---")
    bottom_col1, bottom_col2 = st.columns(2)

    with bottom_col1:
        st.markdown("### ⏰ 24小时时段分布")
        hourly_dash = (
            filtered_df.groupby("H")
            .agg(记录数=("uid", "count"), 完成率=("finish", "mean"), 点赞率=("like", "mean"))
            .reset_index()
        )
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Bar(
                x=hourly_dash["H"], y=hourly_dash["记录数"],
                name="记录数", marker_color="steelblue", opacity=0.7,
            ),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=hourly_dash["H"], y=hourly_dash["完成率"] * 100,
                name="完成率(%)", mode="lines+markers",
                line=dict(color="green", width=2),
            ),
            secondary_y=True,
        )
        fig.update_layout(
            title="24小时分布", height=350, hovermode="x unified",
            margin=dict(t=40, b=10), xaxis=dict(dtick=2),
        )
        fig.update_yaxes(title_text="记录数", secondary_y=False)
        fig.update_yaxes(title_text="比率(%)", secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)

    with bottom_col2:
        st.markdown("### 📉 时长分布 (按Channel)")
        channel_list = sorted(filtered_df["channel"].unique())
        fig = go.Figure()
        colors = ["steelblue", "coral", "seagreen", "purple"]
        for i, ch in enumerate(channel_list[:4]):
            ch_data = filtered_df[filtered_df["channel"] == ch]["duration_time"].clip(0, 30)
            fig.add_trace(go.Violin(
                y=ch_data, name=f"Channel {ch}",
                box_visible=True, meanline_visible=True,
                line_color=colors[i % len(colors)],
                fillcolor=colors[i % len(colors)],
                opacity=0.6,
            ))
        fig.update_layout(
            title="各Channel视频时长分布 (≤30s)",
            height=350, margin=dict(t=40, b=10),
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
    st.sidebar.markdown("## 🎵 抖音数据分析平台")
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
        city_counts = df["user_city"].value_counts()
        selected_cities = city_counts.head(top_n).index.tolist()
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

    # 对需要原始数据的图表，构造轻量筛选 df
    filtered_df = filter_raw_df(df, (start_date, end_date), selected_cities, selected_channels)

    # 数据概览指标
    st.sidebar.markdown(f"### 📊 当前筛选概览")
    st.sidebar.metric("总记录数", f"{len(filtered_df):,}")
    st.sidebar.metric("唯一用户", f"{filtered_df['uid'].nunique():,}")
    st.sidebar.metric("唯一作者", f"{filtered_df['author_id'].nunique():,}")
    st.sidebar.metric("唯一作品", f"{filtered_df['item_id'].nunique():,}")

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

                    # 构建当前数据上下文
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
- 总记录数: {len(filtered_df):,}
- 唯一用户数: {filtered_df['uid'].nunique():,}
- 唯一作者数: {filtered_df['author_id'].nunique():,}
- 唯一作品数: {filtered_df['item_id'].nunique():,}
- 日期范围: {start_date} ~ {end_date}
- 城市数量: {filtered_df['user_city'].nunique()}
- Channel 数量: {filtered_df['channel'].nunique()}

【整体行为指标】
- 平均完成率: {filtered_df['finish'].mean():.2%}
- 平均点赞率: {filtered_df['like'].mean():.2%}
- 平均视频时长: {filtered_df['duration_time'].mean():.1f}秒
- 中位数时长: {filtered_df['duration_time'].median():.0f}秒
- 人均观看次数: {len(filtered_df)/max(filtered_df['uid'].nunique(), 1):.1f}

【Top5 城市记录数】
{top5_cities}

【工作日 vs 周末】
- 工作日记录数: {len(filtered_df[filtered_df['is_weekend']==0]):,}
- 周末记录数: {len(filtered_df[filtered_df['is_weekend']==1]):,}
- 工作日平均完成率: {filtered_df[filtered_df['is_weekend']==0]['finish'].mean():.2%}
- 周末平均完成率: {filtered_df[filtered_df['is_weekend']==1]['finish'].mean():.2%}

【24小时活跃度】
- 最活跃时段 (H): {filtered_df.groupby('H').size().idxmax()}点 ({filtered_df.groupby('H').size().max():,}条)
- 凌晨(0-6)完成率: {filtered_df[filtered_df['H'].between(0,6)]['finish'].mean():.2%}
- 黄金时段(18-23)完成率: {filtered_df[filtered_df['H'].between(18,23)]['finish'].mean():.2%}

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
    st.title("🎵 抖音用户行为数据分析平台")
    st.markdown(
        f"**当前筛选**: {len(filtered_df):,} 条记录 | "
        f"{filtered_df['uid'].nunique():,} 用户 | "
        f"{filtered_df['author_id'].nunique():,} 作者 | "
        f"{date_range[0] if isinstance(date_range, tuple) else date_range} ~ "
        f"{date_range[1] if isinstance(date_range, tuple) else date_range}"
    )

    # ---- 5f. KPI 指标卡 ----
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    n_users_f = max(filtered_df["uid"].nunique(), 1)
    n_days_f = max((end_date - start_date).days, 1)
    with k1:
        st.metric("📈 平均完成率", f"{filtered_df['finish'].mean():.1%}")
    with k2:
        st.metric("❤️ 平均点赞率", f"{filtered_df['like'].mean():.1%}")
    with k3:
        st.metric("⏱️ 平均时长(s)", f"{filtered_df['duration_time'].mean():.1f}")
    with k4:
        st.metric("👤 人均观看", f"{len(filtered_df)/n_users_f:.1f}")
    with k5:
        st.metric("🎬 日均记录", f"{len(filtered_df)/n_days_f:,.0f}")
    with k6:
        st.metric("📹 作品数", f"{filtered_df['item_id'].nunique():,}")

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
        render_dashboard(filtered_df, daily_stats_f, heatmap_data_f)

    # ---- 5i. 页脚 ----
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #888; padding: 10px;'>"
        "🎵 抖音用户行为数据分析平台 | 课程大作业2 | Streamlit + Plotly | 性能优化版"
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
