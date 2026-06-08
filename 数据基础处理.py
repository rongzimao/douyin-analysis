# -*- coding: utf-8 -*-
"""
课程大作业2 - Python数据分析
抖音(Douyin)用户行为数据分析

数据集: douyin_dataset (1).csv
约170万条记录，包含用户观看抖音视频的行为数据

分析维度:
  1. 用户数据可视化 — 用户城市分布/观看数量/观看商品数量/完成率/点赞率
  2. 作者数据可视化 — 作品平均时长/最受欢迎作品/最活跃作者/去重作者数
  3. 作品数据可视化 — 作品时长分布/作品数量/每日作品数/作品点赞数
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import font_manager
import warnings
import os

warnings.filterwarnings('ignore')

# ============================================================
# 0. 全局设置 — 中文字体 & 图表样式
# ============================================================
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = 'analysis_output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("抖音用户行为数据分析 — 课程大作业2")
print("=" * 60)

# ============================================================
# 1. 数据读取与预处理
# ============================================================
print("\n[1/4] 读取数据...")
df = pd.read_csv('douyin_dataset (1).csv')
print(f"  原始数据量: {len(df):,} 条记录, {df.shape[1]} 个字段")

# 删除无用的索引列
if 'Unnamed: 0' in df.columns:
    df.drop(columns=['Unnamed: 0'], inplace=True)

# 转换时间字段
df['real_time'] = pd.to_datetime(df['real_time'])
df['date'] = pd.to_datetime(df['date'])

# 添加辅助列
df['day_of_week'] = df['date'].dt.dayofweek
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

print(f"  字段列表: {df.columns.tolist()}")
print(f"  时间范围: {df['date'].min().date()} ~ {df['date'].max().date()}")
print(f"  唯一用户数: {df['uid'].nunique():,}")
print(f"  唯一作者数: {df['author_id'].nunique():,}")
print(f"  唯一作品数: {df['item_id'].nunique():,}")
print(f"  唯一城市数: {df['user_city'].nunique():,}")

# ============================================================
# 2. 用户数据分析 (User Analysis)
# ============================================================
print("\n[2/4] 用户数据分析...")

# --- 用户级聚合 ---
user_stats = df.groupby('uid').agg(
    watch_count=('item_id', 'count'),         # 观看数量
    item_count=('item_id', 'nunique'),        # 观看不同商品(作品)数量
    finish_rate=('finish', 'mean'),            # 完成率
    like_rate=('like', 'mean'),               # 点赞率
    avg_duration=('duration_time', 'mean'),   # 平均观看时长
    user_city=('user_city', 'first'),         # 用户城市
).reset_index()

print(f"  用户平均观看次数: {user_stats['watch_count'].mean():.1f}")
print(f"  用户平均观看不同作品数: {user_stats['item_count'].mean():.1f}")
print(f"  整体完成率: {df['finish'].mean():.2%}")
print(f"  整体点赞率: {df['like'].mean():.3%}")

# --- Fig.1: 用户观看次数分布 ---
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle('抖音用户行为分析 — 用户维度', fontsize=18, fontweight='bold', y=0.98)

# 1.1 用户观看次数分布 (对数坐标)
ax = axes[0, 0]
watch_counts = user_stats['watch_count']
bins = np.logspace(np.log10(1), np.log10(watch_counts.max()), 50)
ax.hist(watch_counts, bins=bins, color='steelblue', edgecolor='white', alpha=0.85)
ax.set_xscale('log')
ax.set_xlabel('观看次数 (log)')
ax.set_ylabel('用户数量')
ax.set_title('用户观看次数分布')
ax.axvline(watch_counts.median(), color='red', linestyle='--', label=f'中位数: {watch_counts.median():.0f}')
ax.legend()

# 1.2 用户观看不同作品数分布
ax = axes[0, 1]
item_counts = user_stats['item_count']
bins = np.logspace(np.log10(1), np.log10(item_counts.max()), 50)
ax.hist(item_counts, bins=bins, color='darkorange', edgecolor='white', alpha=0.85)
ax.set_xscale('log')
ax.set_xlabel('观看不同作品数 (log)')
ax.set_ylabel('用户数量')
ax.set_title('用户观看不同作品数分布')
ax.axvline(item_counts.median(), color='red', linestyle='--', label=f'中位数: {item_counts.median():.0f}')
ax.legend()

# 1.3 用户城市 Top15
ax = axes[0, 2]
top_cities = df['user_city'].value_counts().head(15)
colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(top_cities)))
bars = ax.barh(range(len(top_cities)), top_cities.values, color=colors[::-1])
ax.set_yticks(range(len(top_cities)))
ax.set_yticklabels([f'城市 {int(c)}' for c in top_cities.index])
ax.set_xlabel('观看记录数')
ax.set_title('用户城市分布 Top15')
ax.invert_yaxis()

# 1.4 用户完成率分布
ax = axes[1, 0]
ax.hist(user_stats['finish_rate'].clip(0, 1), bins=40, color='seagreen', edgecolor='white', alpha=0.85)
ax.set_xlabel('完成率')
ax.set_ylabel('用户数量')
ax.set_title('用户观看完成率分布')
ax.axvline(user_stats['finish_rate'].mean(), color='red', linestyle='--', label=f'均值: {user_stats["finish_rate"].mean():.2%}')
ax.legend()

# 1.5 用户点赞率分布
ax = axes[1, 1]
ax.hist(user_stats['like_rate'].clip(0, 1), bins=40, color='tomato', edgecolor='white', alpha=0.85)
ax.set_xlabel('点赞率')
ax.set_ylabel('用户数量')
ax.set_title('用户点赞率分布')
ax.axvline(user_stats['like_rate'].mean(), color='red', linestyle='--', label=f'均值: {user_stats["like_rate"].mean():.2%}')
ax.legend()

# 1.6 用户活跃度分层 (根据观看次数)
ax = axes[1, 2]
bins_labels = ['低活跃\n(1-2次)', '中低活跃\n(3-10次)', '中活跃\n(11-50次)', '高活跃\n(51-200次)', '超高活跃\n(>200次)']
bins_edges = [0, 2, 10, 50, 200, float('inf')]
user_stats['activity_level'] = pd.cut(user_stats['watch_count'], bins=bins_edges, labels=bins_labels)
activity_counts = user_stats['activity_level'].value_counts()
colors_pie = ['#ff9999', '#ffcc99', '#99ccff', '#99ff99', '#ff99cc']
wedges, texts, autotexts = ax.pie(
    activity_counts.values, labels=activity_counts.index,
    colors=colors_pie, autopct='%1.1f%%',
    explode=(0.02, 0.02, 0.02, 0.02, 0.08),
    startangle=90
)
for t in autotexts:
    t.set_fontsize(8)
ax.set_title('用户活跃度分层')

plt.tight_layout()
fig.savefig(f'{OUTPUT_DIR}/01_用户分析.png', dpi=150, bbox_inches='tight')
plt.close()
print("  -> 01_用户分析.png 已保存")

# --- Fig.2: 用户城市地理分布热力图(简化版) & 行为交叉分析 ---
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('抖音用户行为分析 — 城市与时间分布', fontsize=16, fontweight='bold')

# 按城市汇总
city_stats = df.groupby('user_city').agg(
    record_count=('uid', 'count'),
    unique_users=('uid', 'nunique'),
    avg_finish=('finish', 'mean'),
    avg_like=('like', 'mean'),
).reset_index().sort_values('record_count', ascending=False)

# 2.1 城市 vs 行为气泡图
ax = axes[0]
top30_cities = city_stats.head(30)
scatter = ax.scatter(
    top30_cities['avg_finish'] * 100,
    top30_cities['avg_like'] * 100,
    s=top30_cities['record_count'] / 1000,
    c=top30_cities['unique_users'],
    cmap='plasma', alpha=0.7, edgecolors='black', linewidth=0.3
)
ax.set_xlabel('平均完成率 (%)')
ax.set_ylabel('平均点赞率 (%)')
ax.set_title('Top30城市用户行为气泡图\n(气泡大小=记录数, 颜色=唯一用户数)')
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('唯一用户数')

# 2.2 每日活跃度趋势
ax = axes[1]
daily_stats = df.groupby('date').agg(
    record_count=('uid', 'count'),
    unique_users=('uid', 'nunique'),
    avg_finish=('finish', 'mean'),
    avg_like=('like', 'mean'),
).reset_index()

ax2_twin = ax.twinx()
ax.plot(daily_stats['date'], daily_stats['record_count'], color='steelblue', alpha=0.8, linewidth=1, label='记录数')
ax.plot(daily_stats['date'], daily_stats['unique_users'], color='darkorange', alpha=0.8, linewidth=1, label='活跃用户数')
ax2_twin.plot(daily_stats['date'], daily_stats['avg_finish'] * 100, color='green', alpha=0.5, linestyle=':', label='完成率(%)')
ax2_twin.plot(daily_stats['date'], daily_stats['avg_like'] * 100, color='red', alpha=0.5, linestyle=':', label='点赞率(%)')
ax.set_xlabel('日期')
ax.set_ylabel('数量')
ax2_twin.set_ylabel('比率 (%)')
ax.set_title('每日活跃度与行为趋势')
ax.legend(loc='upper left')
ax2_twin.legend(loc='upper right')
ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
fig.savefig(f'{OUTPUT_DIR}/02_城市与时间分析.png', dpi=150, bbox_inches='tight')
plt.close()
print("  -> 02_城市与时间分析.png 已保存")

# ============================================================
# 3. 作者数据分析 (Author Analysis)
# ============================================================
print("\n[3/4] 作者数据分析...")

# --- 作者级聚合 ---
author_stats = df.groupby('author_id').agg(
    video_count=('item_id', 'nunique'),          # 去重作品数
    total_views=('uid', 'count'),                 # 总观看次数
    avg_duration=('duration_time', 'mean'),       # 作品平均时长
    avg_finish_rate=('finish', 'mean'),           # 平均完成率
    avg_like_rate=('like', 'mean'),               # 平均点赞率
    total_likes=('like', 'sum'),                  # 总点赞数
).reset_index()

# --- 作品级聚合 (用于找最受欢迎作品) ---
item_stats = df.groupby('item_id').agg(
    view_count=('uid', 'count'),
    total_likes=('like', 'sum'),
    like_rate=('like', 'mean'),
    finish_rate=('finish', 'mean'),
    avg_duration=('duration_time', 'mean'),
    author_id=('author_id', 'first'),
).reset_index().sort_values('total_likes', ascending=False)

print(f"  唯一作者数: {len(author_stats):,}")
print(f"  作者平均作品数: {author_stats['video_count'].mean():.1f}")
print(f"  作者人均观看: {author_stats['total_views'].mean():.1f}")

# --- Fig.3: 作者维度分析 ---
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle('抖音用户行为分析 — 作者维度', fontsize=18, fontweight='bold', y=0.98)

# 3.1 作者作品数分布 (对数)
ax = axes[0, 0]
vc = author_stats['video_count']
bins = np.logspace(np.log10(1), np.log10(vc.max()), 50)
ax.hist(vc, bins=bins, color='mediumpurple', edgecolor='white', alpha=0.85)
ax.set_xscale('log')
ax.set_xlabel('作品数 (log)')
ax.set_ylabel('作者数量')
ax.set_title('作者发布作品数分布')
ax.axvline(vc.median(), color='red', linestyle='--', label=f'中位数: {vc.median():.0f}')
ax.legend()

# 3.2 作者平均作品时长分布
ax = axes[0, 1]
durations = author_stats['avg_duration'].clip(0, 60)
ax.hist(durations, bins=50, color='teal', edgecolor='white', alpha=0.85)
ax.set_xlabel('平均作品时长 (秒)')
ax.set_ylabel('作者数量')
ax.set_title('作者平均作品时长分布')
ax.axvline(author_stats['avg_duration'].median(), color='red', linestyle='--', label=f'中位数: {author_stats["avg_duration"].median():.1f}s')
ax.legend()

# 3.3 最受欢迎作品 Top15
ax = axes[0, 2]
top_items = item_stats.head(15)
colors = plt.cm.Reds(np.linspace(0.3, 0.95, len(top_items)))
bars = ax.barh(range(len(top_items)), top_items['total_likes'].values, color=colors[::-1])
ax.set_yticks(range(len(top_items)))
ax.set_yticklabels([f'Item {i}' for i in top_items['item_id'].values])
ax.set_xlabel('总点赞数')
ax.set_title('最受欢迎作品 Top15 (按点赞数)')
ax.invert_yaxis()

# 3.4 最活跃作者 Top15 (按作品数)
ax = axes[1, 0]
top_authors_by_videos = author_stats.nlargest(15, 'video_count')
colors = plt.cm.Oranges(np.linspace(0.3, 0.95, len(top_authors_by_videos)))
bars = ax.barh(range(len(top_authors_by_videos)), top_authors_by_videos['video_count'].values, color=colors[::-1])
ax.set_yticks(range(len(top_authors_by_videos)))
ax.set_yticklabels([f'Author {int(a)}' for a in top_authors_by_videos['author_id'].values])
ax.set_xlabel('作品数量')
ax.set_title('最活跃作者 Top15 (按去重作品数)')
ax.invert_yaxis()

# 3.5 最受欢迎作者 Top15 (按总点赞数)
ax = axes[1, 1]
top_authors_by_likes = author_stats.nlargest(15, 'total_likes')
colors = plt.cm.Reds(np.linspace(0.3, 0.95, len(top_authors_by_likes)))
bars = ax.barh(range(len(top_authors_by_likes)), top_authors_by_likes['total_likes'].values, color=colors[::-1])
ax.set_yticks(range(len(top_authors_by_likes)))
ax.set_yticklabels([f'Author {int(a)}' for a in top_authors_by_likes['author_id'].values])
ax.set_xlabel('总点赞数')
ax.set_title('最受欢迎作者 Top15 (按总点赞数)')
ax.invert_yaxis()

# 3.6 作者观看数 vs 作品数散点图
ax = axes[1, 2]
sample_authors = author_stats[author_stats['video_count'] <= 100]
ax.hexbin(
    sample_authors['video_count'],
    np.log10(sample_authors['total_views'].clip(lower=1)),
    gridsize=40, cmap='YlOrRd', mincnt=1
)
ax.set_xlabel('作品数')
ax.set_ylabel('总观看次数 (log10)')
ax.set_title('作者作品数 vs 总观看次数\n(热力图, 作品数≤100)')
cbar = plt.colorbar(ax.collections[0], ax=ax)
cbar.set_label('作者数量')

plt.tight_layout()
fig.savefig(f'{OUTPUT_DIR}/03_作者分析.png', dpi=150, bbox_inches='tight')
plt.close()
print("  -> 03_作者分析.png 已保存")

# ============================================================
# 4. 作品数据分析 (Content/Item Analysis)
# ============================================================
print("\n[4/4] 作品数据分析...")

# --- Fig.4: 作品维度分析 ---
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle('抖音用户行为分析 — 作品维度', fontsize=18, fontweight='bold', y=0.98)

# 4.1 作品时长分布
ax = axes[0, 0]
durations_clipped = df['duration_time'].clip(0, 60)
ax.hist(durations_clipped, bins=60, color='cornflowerblue', edgecolor='white', alpha=0.85)
ax.set_xlabel('时长 (秒)')
ax.set_ylabel('记录数')
ax.set_title('视频时长分布 (≤60s)')
ax.axvline(df['duration_time'].median(), color='red', linestyle='--', label=f'中位数: {df["duration_time"].median():.0f}s')
ax.axvline(df['duration_time'].mean(), color='orange', linestyle='--', label=f'均值: {df["duration_time"].mean():.1f}s')
ax.legend()

# 4.2 作品时长分布 - 全范围
ax = axes[0, 1]
all_durations = df['duration_time']
bins = np.logspace(np.log10(1), np.log10(all_durations.max()), 60) if all_durations.min() > 0 else np.linspace(0, all_durations.max(), 60)
ax.hist(all_durations[all_durations > 0], bins=50, color='slateblue', edgecolor='white', alpha=0.85)
ax.set_xlabel('时长 (秒)')
ax.set_ylabel('记录数')
ax.set_title('视频时长分布 (全范围)')
ax.axvline(all_durations.median(), color='red', linestyle='--', label=f'中位数: {all_durations.median():.0f}s')
ax.legend()

# 4.3 每日作品(观看记录)数趋势
ax = axes[0, 2]
daily_items = df.groupby('date').agg(
    total_records=('uid', 'count'),
    unique_items=('item_id', 'nunique'),
).reset_index()

ax.plot(daily_items['date'], daily_items['total_records'], color='steelblue', linewidth=1.5, label='总记录数')
ax.plot(daily_items['date'], daily_items['unique_items'], color='coral', linewidth=1.5, label='去重作品数')
ax.set_xlabel('日期')
ax.set_ylabel('数量')
ax.set_title('每日观看记录与作品数趋势')
ax.legend()
ax.tick_params(axis='x', rotation=45)

# 4.4 每小时时段分析
ax = axes[1, 0]
hourly_stats = df.groupby('H').agg(
    record_count=('uid', 'count'),
    avg_finish=('finish', 'mean'),
    avg_like=('like', 'mean'),
).reset_index()

ax2 = ax.twinx()
ax.bar(hourly_stats['H'], hourly_stats['record_count'], color='steelblue', alpha=0.7, width=0.8)
ax2.plot(hourly_stats['H'], hourly_stats['avg_finish'] * 100, 'o-', color='green', linewidth=2, label='完成率(%)')
ax2.plot(hourly_stats['H'], hourly_stats['avg_like'] * 100, 's-', color='red', linewidth=2, label='点赞率(%)')
ax.set_xlabel('小时')
ax.set_ylabel('记录数')
ax2.set_ylabel('比率 (%)')
ax.set_title('24小时活跃度分布')
ax.set_xticks(range(0, 24, 2))
ax2.legend(loc='upper right')

# 4.5 作品点赞分布
ax = axes[1, 1]
like_by_item = df.groupby('item_id')['like'].sum()
like_bins = np.arange(0, like_by_item.max() + 2, 1)
ax.hist(like_by_item.clip(0, 10), bins=np.arange(-0.5, 10.5, 1), color='tomato', edgecolor='white', alpha=0.85)
ax.set_xlabel('点赞数')
ax.set_ylabel('作品数量')
ax.set_title('作品点赞数分布 (0-10赞)')
ax.set_xticks(range(0, 11))

# 4.6 Channel 分布
ax = axes[1, 2]
channel_stats = df.groupby('channel').agg(
    count=('uid', 'count'),
    avg_finish=('finish', 'mean'),
    avg_like=('like', 'mean'),
).reset_index()
channel_stats['channel_label'] = channel_stats['channel'].map({0: 'Channel 0', 1: 'Channel 1', 2: 'Channel 2', 3: 'Channel 3'})

x = range(len(channel_stats))
width = 0.35
bars1 = ax.bar([i - width/2 for i in x], channel_stats['avg_finish'] * 100, width, color='seagreen', alpha=0.8, label='平均完成率(%)')
bars2 = ax.bar([i + width/2 for i in x], channel_stats['avg_like'] * 100, width, color='tomato', alpha=0.8, label='平均点赞率(%)')
ax.set_xticks(x)
ax.set_xticklabels(channel_stats['channel_label'])
ax.set_ylabel('比率 (%)')
ax.set_title('各Channel完成率与点赞率对比')
ax.legend()

# 在柱状图上添加数值
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, f'{bar.get_height():.1f}%', ha='center', fontsize=8)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, f'{bar.get_height():.1f}%', ha='center', fontsize=8)

plt.tight_layout()
fig.savefig(f'{OUTPUT_DIR}/04_作品分析.png', dpi=150, bbox_inches='tight')
plt.close()
print("  -> 04_作品分析.png 已保存")

# ============================================================
# 5. 综合仪表盘 (Dashboard Summary)
# ============================================================
print("\n生成综合仪表盘...")

fig = plt.figure(figsize=(20, 14))
fig.suptitle('抖音用户行为分析 — 综合数据仪表盘', fontsize=22, fontweight='bold', y=0.97)

# 5.1 关键指标卡片 (左上)
ax_metrics = fig.add_axes([0.04, 0.82, 0.35, 0.13])
ax_metrics.axis('off')
metrics_text = (
    f"📊 数据总览\n"
    f"总记录数: {len(df):,}  |  唯一用户: {df['uid'].nunique():,}\n"
    f"唯一作者: {df['author_id'].nunique():,}  |  唯一作品: {df['item_id'].nunique():,}\n"
    f"时间跨度: {df['date'].min().date()} ~ {df['date'].max().date()}\n"
    f"完成率: {df['finish'].mean():.2%}  |  点赞率: {df['like'].mean():.3%}"
)
ax_metrics.text(0.5, 0.5, metrics_text, transform=ax_metrics.transAxes,
                fontsize=12, verticalalignment='center', horizontalalignment='center',
                fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

# 5.2 每日趋势 (中上)
ax_daily = fig.add_axes([0.42, 0.82, 0.55, 0.13])
daily_items['weekday'] = daily_items['date'].dt.day_name()
ax_daily.fill_between(daily_items['date'], daily_items['total_records'], alpha=0.5, color='steelblue')
ax_daily.plot(daily_items['date'], daily_items['total_records'], color='steelblue', linewidth=1)
ax_daily.set_title('每日总记录数趋势', fontsize=12)
ax_daily.tick_params(axis='x', rotation=45, labelsize=8)

# 5.3 用户Top城市 (左下)
ax_cities = fig.add_axes([0.04, 0.42, 0.30, 0.35])
top20 = city_stats.head(20)
ax_cities.barh(range(len(top20)), top20['record_count'].values, color=plt.cm.Blues(np.linspace(0.3, 0.9, len(top20))))
ax_cities.set_yticks(range(len(top20)))
ax_cities.set_yticklabels([f'City {int(c)}' for c in top20['user_city'].values], fontsize=8)
ax_cities.set_title('Top20 用户城市', fontsize=11)
ax_cities.invert_yaxis()

# 5.4 时段热力图 (中下)
ax_hour = fig.add_axes([0.38, 0.42, 0.28, 0.35])
# 按小时和星期几分组
df['weekday'] = df['date'].dt.dayofweek
heatmap_data = df.pivot_table(index='H', columns='weekday', values='uid', aggfunc='count')
weekday_labels = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
heatmap_data.columns = weekday_labels[:len(heatmap_data.columns)]
im = ax_hour.imshow(heatmap_data.values, aspect='auto', cmap='YlOrRd', interpolation='bilinear')
ax_hour.set_xticks(range(len(heatmap_data.columns)))
ax_hour.set_xticklabels(heatmap_data.columns, fontsize=9)
ax_hour.set_yticks(range(0, 24, 2))
ax_hour.set_ylabel('小时', fontsize=10)
ax_hour.set_title('时段 × 星期 活跃度热力图', fontsize=11)
plt.colorbar(im, ax=ax_hour, label='记录数')

# 5.5 作者生态分布 (右下)
ax_author = fig.add_axes([0.70, 0.42, 0.27, 0.35])
author_bins = [0, 1, 2, 5, 10, 20, 50, float('inf')]
author_labels = ['1', '2', '3-5', '6-10', '11-20', '21-50', '>50']
author_stats['group'] = pd.cut(author_stats['video_count'], bins=author_bins, labels=author_labels)
group_counts = author_stats['group'].value_counts()
colors_pie = plt.cm.Set3(np.linspace(0, 1, len(group_counts)))
wedges, texts, autotexts = ax_author.pie(
    group_counts.values, labels=group_counts.index, colors=colors_pie,
    autopct='%1.1f%%', startangle=140, pctdistance=0.8
)
for t in autotexts:
    t.set_fontsize(8)
ax_author.set_title('作者发布作品数分层', fontsize=11)

# 5.6 作品时长箱线图 (底部)
ax_box = fig.add_axes([0.04, 0.04, 0.93, 0.33])
channel_labels = ['Channel 0', 'Channel 1', 'Channel 2', 'Channel 3']
box_data = [df[df['channel'] == c]['duration_time'].clip(0, 30).values for c in sorted(df['channel'].unique())]
bp = ax_box.boxplot(box_data, labels=channel_labels, patch_artist=True, showfliers=False,
                     boxprops=dict(facecolor='lightblue', alpha=0.7),
                     medianprops=dict(color='red', linewidth=2))
ax_box.set_ylabel('时长 (秒)', fontsize=11)
ax_box.set_title('各Channel视频时长分布 (≤30s, 去异常值)', fontsize=12)
# 添加均值点
for i, data in enumerate(box_data):
    ax_box.scatter(i + 1, np.mean(data), color='darkred', s=80, zorder=5, marker='D', label='均值' if i == 0 else '')

# 添加完成率统计在箱线图旁边
finish_by_channel = df.groupby('channel')['finish'].mean() * 100
for i, (_, row) in enumerate(finish_by_channel.items()):
    ax_box.text(i + 1.15, 27, f'完成率\n{row:.1f}%', fontsize=9, ha='center', color='green')

ax_box.legend(loc='upper right')

fig.savefig(f'{OUTPUT_DIR}/05_综合仪表盘.png', dpi=150, bbox_inches='tight')
plt.close()
print("  -> 05_综合仪表盘.png 已保存")

# ============================================================
# 6. 数据摘要输出
# ============================================================
print("\n" + "=" * 60)
print("分析完成! 数据摘要:")
print("=" * 60)
print(f"""
┌─────────────────────────────────────────────┐
│          抖音用户行为分析 - 数据摘要          │
├─────────────────────────────────────────────┤
│ 数据规模                                     │
│   • 总记录数: {len(df):>12,}                │
│   • 唯一用户: {df['uid'].nunique():>12,}                │
│   • 唯一作者: {df['author_id'].nunique():>12,}                │
│   • 唯一作品: {df['item_id'].nunique():>12,}                │
├─────────────────────────────────────────────┤
│ 用户行为指标                                 │
│   • 人均观看次数: {user_stats['watch_count'].mean():>10.1f}            │
│   • 人均观看不同作品: {user_stats['item_count'].mean():>7.1f}          │
│   • 整体完成率: {df['finish'].mean():>13.2%}              │
│   • 整体点赞率: {df['like'].mean():>13.3%}              │
├─────────────────────────────────────────────┤
│ 作品维度                                     │
│   • 平均视频时长: {df['duration_time'].mean():>10.1f}s            │
│   • 视频时长中位数: {df['duration_time'].median():>8.0f}s              │
│   • 每日平均记录数: {daily_items['total_records'].mean():>8.0f}          │
│   • 每日平均作品数: {daily_items['unique_items'].mean():>8.0f}          │
├─────────────────────────────────────────────┤
│ 作者维度                                     │
│   • 人均发布作品: {author_stats['video_count'].mean():>10.1f}            │
│   • 发布最多作品作者: {author_stats['video_count'].max():>7.0f} 个        │
│   • 单作品最多点赞: {item_stats['total_likes'].max():>7.0f}             │
├─────────────────────────────────────────────┤
│ 分析图表已保存至: {OUTPUT_DIR}/              │
│   • 01_用户分析.png                          │
│   • 02_城市与时间分析.png                     │
│   • 03_作者分析.png                          │
│   • 04_作品分析.png                          │
│   • 05_综合仪表盘.png                        │
└─────────────────────────────────────────────┘
""")

print("全部完成! 共生成 5 张分析图表。")
