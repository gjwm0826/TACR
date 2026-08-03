"""
Phase A7: 图表生成脚本
功能：从GPU指标日志和实验结果中生成论文所需的图表
运行：python generate_figures.py
输出：多个PNG图表文件
论文位置：各图表对应论文中的Figure/Table

依赖：matplotlib, numpy
"""

import json
import os
import glob
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

# ============================================================
# 配置
# ============================================================
BASE_DIR = r'/root/autodl-tmp/TACR/'
DATA_DIR = os.path.join(BASE_DIR, 'org', 'myrag', 'summary', 'extractive', 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'figures')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 设置字体（避免中文乱码）
# 替换为以下配置
plt.rcParams['font.family'] = 'serif'
# 优先尝试 Times New Roman，若无则自动使用 Linux 自带的高质量衬线体（避免报错）
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Liberation Serif', 'serif']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11
plt.rcParams['legend.fontsize'] = 10

# 颜色方案
COLORS = {
    'TACR': '#2196F3',
    'FullRAG': '#4CAF50',
    'NoRAG': '#FF5722',
    'BM25': '#FF9800',
    'COMI': '#9C27B0',
    'PISCO': '#795548',
    'xRAG': '#607D8B',
    'Random': '#9E9E9E',
    'FirstK': '#CDDC39'
}


def analyze_gpu_log(filepath):
    """分析GPU指标日志"""
    if not os.path.exists(filepath):
        return None

    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if not data or len(data) == 0:
        return None

    samples = data[1:] if len(data) > 1 else data

    input_tokens = [s.get('input_tokens_approx', 0) for s in samples]
    api_latencies = [s.get('api_latency_ms', 0) for s in samples]
    kv_caches = [s.get('estimated_kv_cache_mb', 0) for s in samples]

    return {
        "num_samples": len(data),
        "avg_tokens": statistics.mean(input_tokens) if input_tokens else 0,
        "avg_latency": statistics.mean(api_latencies) if api_latencies else 0,
        "std_latency": statistics.stdev(api_latencies) if len(api_latencies) > 1 else 0,
        "avg_kv_cache": statistics.mean(kv_caches) if kv_caches else 0,
    }


# ============================================================
# Figure 1: K值 vs QA准确率（消融实验）
# ============================================================

def generate_k_ablation_figure():
    """
    生成K值消融实验图。
    X轴: K值 (2,4,6,8,10,12,14,16)
    Y轴: QA准确率
    多条线: NQ, HotpotQA
    """
    # 注意：准确率数据需要从stat_accuracy脚本获取
    # 这里使用token_compression_comparison.py中的数据

    # NQ数据（来自token_compression_comparison.py）
    nq_compression_ratios = [0.05, 0.08, 0.11, 0.14, 0.17, 0.20, 0.23, 0.26]
    nq_accuracy = [0.6918, 0.7464, 0.7870, 0.8175, 0.8381, 0.8520, 0.8580, 0.8671]

    # HotpotQA数据
    hotpot_compression_ratios = [0.05, 0.08, 0.11, 0.14, 0.17, 0.20, 0.23, 0.26]
    hotpot_accuracy = [0.4551, 0.5192, 0.5769, 0.6154, 0.6410, 0.6538, 0.6603, 0.6731]

    # 基线
    nq_fullrag_acc = 0.8882
    nq_norag_acc = 0.6103
    hotpot_fullrag_acc = 0.6346
    hotpot_norag_acc = 0.4872

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # NQ
    ax1.plot(nq_compression_ratios, nq_accuracy, 'o-', color=COLORS['TACR'], linewidth=2, markersize=6, label='TACR')
    ax1.axhline(y=nq_fullrag_acc, color=COLORS['FullRAG'], linestyle='--', linewidth=1.5, label='Full RAG')
    ax1.axhline(y=nq_norag_acc, color=COLORS['NoRAG'], linestyle=':', linewidth=1.5, label='No RAG')
    ax1.set_xlabel('Document Length Ratio')
    ax1.set_ylabel('QA Accuracy')
    ax1.set_title('NQ (Natural Questions)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0.6, 0.95)

    # HotpotQA
    ax2.plot(hotpot_compression_ratios, hotpot_accuracy, 'o-', color=COLORS['TACR'], linewidth=2, markersize=6, label='TACR')
    ax2.axhline(y=hotpot_fullrag_acc, color=COLORS['FullRAG'], linestyle='--', linewidth=1.5, label='Full RAG')
    ax2.axhline(y=hotpot_norag_acc, color=COLORS['NoRAG'], linestyle=':', linewidth=1.5, label='No RAG')
    ax2.set_xlabel('Document Length Ratio')
    ax2.set_ylabel('QA Accuracy')
    ax2.set_title('HotpotQA')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0.4, 0.75)

    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, 'fig_k_ablation_accuracy.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"已生成: {filepath}")


# ============================================================
# Figure 2: K值 vs 延迟 + KV-Cache
# ============================================================

def generate_latency_kvcache_figure():
    """
    生成K值 vs 延迟和KV-Cache的图。
    基于实际GPU指标日志数据。
    """
    recomp_dir = os.path.join(DATA_DIR, 'recomp')
    hotpot_dir = os.path.join(DATA_DIR, 'hotpot')

    # 收集所有K值的日志
    nq_logs = sorted(glob.glob(os.path.join(recomp_dir, 'NQ提取式摘要Llama gpu_metrics_log*选句*.json')))
    hotpot_logs = sorted(glob.glob(os.path.join(hotpot_dir, 'Hotpot提取式摘要Llama gpu_metrics_log*选句*.json')))

    # 解析K值和统计数据
    nq_data = []
    for log_file in nq_logs:
        stats = analyze_gpu_log(log_file)
        if stats:
            fname = os.path.basename(log_file)
            # 尝试从文件名提取K值
            import re
            k_match = re.search(r'选句(\d+)', fname)
            if k_match:
                k = int(k_match.group(1))
                nq_data.append((k, stats))

    hotpot_data = []
    for log_file in hotpot_logs:
        stats = analyze_gpu_log(log_file)
        if stats:
            fname = os.path.basename(log_file)
            k_match = re.search(r'选句(\d+)', fname)
            if k_match:
                k = int(k_match.group(1))
                hotpot_data.append((k, stats))

    nq_data.sort(key=lambda x: x[0])
    hotpot_data.sort(key=lambda x: x[0])

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # NQ 延迟
    if nq_data:
        ks = [d[0] for d in nq_data]
        latencies = [d[1]['avg_latency'] for d in nq_data]
        stds = [d[1]['std_latency'] for d in nq_data]
        axes[0, 0].bar(range(len(ks)), latencies, yerr=stds, color=COLORS['TACR'], alpha=0.7, capsize=3)
        axes[0, 0].set_xticks(range(len(ks)))
        axes[0, 0].set_xticklabels([str(k) for k in ks])
        axes[0, 0].set_xlabel('K (Number of Selected Sentences)')
        axes[0, 0].set_ylabel('API Latency (ms)')
        axes[0, 0].set_title('NQ: Inference Latency vs K')
        axes[0, 0].grid(True, alpha=0.3, axis='y')

    # NQ KV-Cache
    if nq_data:
        ks = [d[0] for d in nq_data]
        kv = [d[1]['avg_kv_cache'] for d in nq_data]
        axes[0, 1].plot(ks, kv, 'o-', color=COLORS['TACR'], linewidth=2, markersize=6)
        axes[0, 1].set_xlabel('K (Number of Selected Sentences)')
        axes[0, 1].set_ylabel('KV-Cache Size (MB)')
        axes[0, 1].set_title('NQ: KV-Cache Size vs K')
        axes[0, 1].grid(True, alpha=0.3)

    # HotpotQA 延迟
    if hotpot_data:
        ks = [d[0] for d in hotpot_data]
        latencies = [d[1]['avg_latency'] for d in hotpot_data]
        stds = [d[1]['std_latency'] for d in hotpot_data]
        axes[1, 0].bar(range(len(ks)), latencies, yerr=stds, color='#FF9800', alpha=0.7, capsize=3)
        axes[1, 0].set_xticks(range(len(ks)))
        axes[1, 0].set_xticklabels([str(k) for k in ks])
        axes[1, 0].set_xlabel('K (Number of Selected Sentences)')
        axes[1, 0].set_ylabel('API Latency (ms)')
        axes[1, 0].set_title('HotpotQA: Inference Latency vs K')
        axes[1, 0].grid(True, alpha=0.3, axis='y')

    # HotpotQA KV-Cache
    if hotpot_data:
        ks = [d[0] for d in hotpot_data]
        kv = [d[1]['avg_kv_cache'] for d in hotpot_data]
        axes[1, 1].plot(ks, kv, 'o-', color='#FF9800', linewidth=2, markersize=6)
        axes[1, 1].set_xlabel('K (Number of Selected Sentences)')
        axes[1, 1].set_ylabel('KV-Cache Size (MB)')
        axes[1, 1].set_title('HotpotQA: KV-Cache Size vs K')
        axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, 'fig_latency_kvcache_vs_k.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"已生成: {filepath}")


# ============================================================
# Figure 3: 方法对比总览
# ============================================================

def generate_method_comparison_figure():
    """
    生成各方法对比图（准确率 vs Token数）。
    需要baseline_comparison_results.json中的数据。
    """
    baseline_path = os.path.join('/root/autodl-tmp/TACR/org/myrag/summary/extractive/', 'baseline_comparison_results.json')

    if not os.path.exists(baseline_path):
        print("警告: baseline_comparison_results.json 不存在，跳过此图")
        print("请先运行 baseline_comparison.py")
        return

    with open(baseline_path, 'r') as f:
        metrics = json.load(f)

    fig, ax = plt.subplots(figsize=(10, 6))

    methods = []
    accuracies = []
    tokens = []
    colors = []

    color_map = {
        'TACR': COLORS['TACR'],
        'FullRAG': COLORS['FullRAG'],
        'NoRAG': COLORS['NoRAG'],
        'BM25_TopK': COLORS['BM25'],
        'COMI_Lite': COLORS['COMI'],
        'PISCO_Inspired': COLORS['PISCO'],
        'xRAG_Inspired': COLORS['xRAG'],
        'Random_K': COLORS['Random'],
        'First_K': COLORS['FirstK']
    }

    for m in metrics:
        methods.append(m['method'])
        accuracies.append(m['accuracy'] * 100)
        tokens.append(m['avg_tokens'])
        colors.append(color_map.get(m['method'], '#999999'))

    scatter = ax.scatter(tokens, accuracies, c=colors, s=150, edgecolors='black', linewidth=0.5, zorder=3)

    # 标注方法名称
    for i, method in enumerate(methods):
        ax.annotate(method, (tokens[i], accuracies[i]),
                   textcoords="offset points", xytext=(5, 5), fontsize=9)

    ax.set_xlabel('Average Input Tokens')
    ax.set_ylabel('QA Accuracy (%)')
    ax.set_title('Method Comparison: Accuracy vs Context Length')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, 'fig_method_comparison.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"已生成: {filepath}")


# ============================================================
# Figure 4: Token压缩比 vs 准确率权衡
# ============================================================

def generate_compression_tradeoff_figure():
    """
    生成压缩率 vs 准确率权衡图。
    复现token_compression_comparison.py中的数据。
    """
    # 数据来自token_compression_comparison.py
    datasets = {
        'NQ (RecompQA)': {
            'compression_ratios': [0.05, 0.08, 0.11, 0.14, 0.17, 0.20, 0.23, 0.26],
            'accuracy': [0.6918, 0.7464, 0.7870, 0.8175, 0.8381, 0.8520, 0.8580, 0.8671],
            'fullrag': 0.8882,
            'norag': 0.6103,
            'color': COLORS['TACR']
        },
        'HotpotQA': {
            'compression_ratios': [0.05, 0.08, 0.11, 0.14, 0.17, 0.20, 0.23, 0.26],
            'accuracy': [0.4551, 0.5192, 0.5769, 0.6154, 0.6410, 0.6538, 0.6603, 0.6731],
            'fullrag': 0.6346,
            'norag': 0.4872,
            'color': '#FF9800'
        }
    }

    fig, ax = plt.subplots(figsize=(10, 6))

    for name, data in datasets.items():
        ax.plot(data['compression_ratios'], data['accuracy'], 'o-',
                color=data['color'], linewidth=2, markersize=6, label=f'TACR ({name})')
        ax.axhline(y=data['fullrag'], color=COLORS['FullRAG'], linestyle='--',
                   linewidth=1, alpha=0.5)
        ax.axhline(y=data['norag'], color=COLORS['NoRAG'], linestyle=':',
                   linewidth=1, alpha=0.5)

    ax.set_xlabel('Document Length Ratio (Compressed / Original)')
    ax.set_ylabel('QA Accuracy')
    ax.set_title('Compression Ratio vs QA Accuracy')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    # 添加Full RAG和No RAG标签
    ax.text(0.27, 0.895, 'Full RAG', fontsize=9, color=COLORS['FullRAG'])
    ax.text(0.27, 0.615, 'No RAG', fontsize=9, color=COLORS['NoRAG'])

    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, 'fig_compression_tradeoff.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"已生成: {filepath}")


# ============================================================
# 主函数
# ============================================================

def main():
    print("TACR 图表生成")
    print("=" * 60)

    generate_k_ablation_figure()
    generate_latency_kvcache_figure()
    generate_compression_tradeoff_figure()
    generate_method_comparison_figure()

    print(f"\n所有图表已保存到: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
