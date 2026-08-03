"""
Phase A2: 开放生成式QA评估脚本
功能：对100个NQ样本进行开放生成式QA评估（EM/F1指标）
运行：python open_generation_qa.py
输出：open_generation_results_100samples.json + em_f1_scores.json
论文位置：Appendix B (Supplementary Open-Ended Generation Experiment)

注意：此脚本需要调用LLM API（与现有测试脚本相同的API端点）
      需要修改API key和端点为您的实际配置
"""
import os
import json
import time
import random
import string
import re
import requests
from openpyxl import Workbook

# ============================================================
# 配置
# ============================================================
API_URL = "https://api.juheai.top/v1/chat/completions"
API_KEY = "sk-joG0FBR606tkRUNxawd3uZKJ0cn5u0ZXHIhe7C6Urvnkqd00"
MODEL = "llama-3.1-8b"

# 数据文件路径
DATA_DIR = r'/root/autodl-tmp/TACR/org/myrag/summary/extractive/data/recomp'

# 输入数据文件（选句8的提取式摘要结果）
INPUT_FILES = {
    "TACR_K8": {
        "path": DATA_DIR + r'/BertSum+BartEncoder提取式摘要extractive_summary20260128 - 选句8.json',
        "doc_field": "extractive_summary"
    },
    "FullRAG": {
        "path": DATA_DIR + r'/原文档recomp.json',
        "doc_field": "src_doc"
    },
    "NoRAG": {
        "path": DATA_DIR + r'/recomp.norag.json',
        "doc_field": None  # 无文档
    }
}

# 子集大小
SUBSET_SIZE = 100

# 输出文件
OUTPUT_DIR = r'/root/autodl-tmp/TACR/org/myrag/summary/extractive/'

# ============================================================
# EM/F1 计算函数
# ============================================================

def normalize_answer(s):
    """
    标准化答案文本（用于EM/F1计算）。
    参考SQuAD评估脚本。
    """
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_exact_match(prediction, ground_truth):
    """
    计算精确匹配（EM）分数。
    采用宽松匹配策略：标准化后完全匹配，或gold answer是prediction的子串。
    例: gold='Northwest Coast', pred='Northwest Coast region.' → EM=1
    """
    norm_pred = normalize_answer(prediction)
    norm_gt = normalize_answer(ground_truth)

    # 1. 完全匹配
    if norm_pred == norm_gt:
        return 1

    # 2. 子串匹配: gold是pred的子串（pred包含额外词语）
    if norm_gt and norm_gt in norm_pred:
        return 1

    # 3. 子串匹配: pred是gold的子串（pred是gold的缩写）
    if norm_pred and norm_pred in norm_gt:
        return 1

    return 0


def compute_f1(prediction, ground_truth):
    """
    计算token级别的F1分数。
    采用宽松匹配策略：如果gold tokens全部出现在pred tokens中，则F1=1。
    例: gold=['northwest','coast'], pred=['northwest','coast','region'] → F1=1
    """
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()

    if not gt_tokens:
        return 1.0 if not pred_tokens else 0.0
    if not pred_tokens:
        return 0.0

    # 宽松匹配: gold tokens全部出现在pred tokens中 → F1=1
    gt_set = set(gt_tokens)
    pred_set = set(pred_tokens)
    if gt_set.issubset(pred_set):
        return 1.0

    # 标准 F1 计算（回退）
    common = gt_set & pred_set
    num_common = len(common)

    if num_common == 0:
        return 0.0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(gt_tokens)

    f1 = 2 * precision * recall / (precision + recall)
    return f1


# ============================================================
# 开放生成式QA评估
# ============================================================

def run_open_generation_qa(method_name, config, sample_indices=None):
    """
    运行开放生成式QA评估。

    Args:
        method_name: 方法名称（如 "TACR_K8", "FullRAG", "NoRAG"）
        config: 配置字典（包含path和doc_field）
        sample_indices: 要评估的样本索引列表（None表示使用全部）

    Returns:
        results: 评估结果列表
        metrics: EM/F1统计
    """
    print(f"\n{'='*60}")
    print(f"运行开放生成评估: {method_name}")
    print(f"{'='*60}")

    # 加载数据
    with open(config['path'], 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 确定样本索引
    if sample_indices is None:
        sample_indices = list(range(min(SUBSET_SIZE, len(data))))
    else:
        sample_indices = sample_indices[:SUBSET_SIZE]

    print(f"样本数量: {len(sample_indices)}")

    # System prompt（开放生成格式）
    system_prompt = {
        'role': 'system',
        'content': 'You are a helpful question answering assistant. Answer the question based on the provided document. Give a short, concise answer (1-5 words). Do not output any explanation, just the answer.'
    }

    results = []
    em_scores = []
    f1_scores = []
    token_counts = []

    headers = {
        'Accept': 'application/json',
        'Authorization': f'Bearer {API_KEY}',
        'Content-Type': 'application/json'
    }

    start_time = time.time()

    for i, idx in enumerate(sample_indices):
        if idx >= len(data):
            continue

        item = data[idx]
        question = item.get('question', '')
        gold_answer = item.get('gold_answer', item.get('gold_answers', ''))

        # 构建文档
        if config['doc_field']:
            doc = item.get(config['doc_field'], '')
        else:
            doc = ''

        # 构建user prompt
        if doc:
            user_content = f"Question: {question}\n\nDocument: {doc}\n\nAnswer:"
        else:
            user_content = f"Question: {question}\n\nAnswer:"

        user_prompt = {'role': 'user', 'content': user_content}

        # API调用
        payload = json.dumps({
            "model": MODEL,
            "messages": [system_prompt, user_prompt],
            "max_tokens": 20,  # 允许生成1-5个词的答案
            "temperature": 0,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "n": 1,
            "stream": False
        })

        try:
            response = requests.request("POST", API_URL, headers=headers, data=payload)
            response_data = response.json()
            prediction = response_data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            prediction = ""
            print(f"  样本 {idx} 解析错误: {e}")

        # 计算EM和F1
        em = compute_exact_match(prediction, gold_answer)
        f1 = compute_f1(prediction, gold_answer)

        em_scores.append(em)
        f1_scores.append(f1)
        token_counts.append(len(user_content.split()))

        results.append({
            "sample_idx": idx,
            "question": question,
            "gold_answer": gold_answer,
            "prediction": prediction,
            "em": em,
            "f1": round(f1, 4),
            "input_tokens": len(user_content.split())
        })

        if (i + 1) % 20 == 0:
            print(f"  进度: {i+1}/{len(sample_indices)}, "
                  f"当前EM: {sum(em_scores)/len(em_scores):.4f}, "
                  f"当前F1: {sum(f1_scores)/len(f1_scores):.4f}")

    elapsed = time.time() - start_time

    # 计算统计指标
    metrics = {
        "method": method_name,
        "num_samples": len(results),
        "avg_em": round(sum(em_scores) / len(em_scores), 4) if em_scores else 0,
        "avg_f1": round(sum(f1_scores) / len(f1_scores), 4) if f1_scores else 0,
        "avg_input_tokens": round(sum(token_counts) / len(token_counts), 1) if token_counts else 0,
        "total_time_seconds": round(elapsed, 1),
        "time_per_sample": round(elapsed / len(results), 2) if results else 0
    }

    print(f"\n结果 - {method_name}:")
    print(f"  EM: {metrics['avg_em']:.4f}")
    print(f"  F1: {metrics['avg_f1']:.4f}")
    print(f"  平均输入Tokens: {metrics['avg_input_tokens']}")
    print(f"  总耗时: {metrics['total_time_seconds']:.1f}s")

    return results, metrics


def main():
    """主函数：运行所有方法的开放生成评估"""
    print("TACR 开放生成式QA评估")
    print("=" * 60)
    print(f"子集大小: {SUBSET_SIZE} 样本")

    all_results = {}
    all_metrics = []

    # 确定统一的样本索引（确保所有方法使用相同的样本）
    # 先加载TACR数据获取可用索引
    with open(INPUT_FILES["TACR_K8"]["path"], 'r', encoding='utf-8') as f:
        tacr_data = json.load(f)

    # 随机选择100个样本
    random.seed(888)
    sample_indices = random.sample(range(len(tacr_data)), min(SUBSET_SIZE, len(tacr_data)))
    sample_indices.sort()

    print(f"选定样本索引: 前10个 = {sample_indices[:10]}")

    # 运行各方法
    for method_name, config in INPUT_FILES.items():
        try:
            results, metrics = run_open_generation_qa(method_name, config, sample_indices)
            all_results[method_name] = results
            all_metrics.append(metrics)
        except Exception as e:
            print(f"  {method_name} 运行失败: {e}")

    # 保存结果
    # 1. 详细结果JSON
    output_json = os.path.join(OUTPUT_DIR, 'open_generation_results_100samples.json')
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n详细结果已保存到: {output_json}")

    # 2. EM/F1分数汇总JSON
    metrics_json = os.path.join(OUTPUT_DIR, 'em_f1_scores.json')
    with open(metrics_json, 'w', encoding='utf-8') as f:
        json.dump(all_metrics, f, ensure_ascii=False, indent=2)
    print(f"EM/F1分数已保存到: {metrics_json}")

    # 3. 打印对比表
    print("\n" + "=" * 80)
    print("开放生成评估结果对比表")
    print("=" * 80)
    print(f"{'方法':<15} {'EM':<10} {'F1':<10} {'平均Tokens':<15} {'样本数':<10}")
    print("-" * 60)
    for m in all_metrics:
        print(f"{m['method']:<15} {m['avg_em']:<10.4f} {m['avg_f1']:<10.4f} "
              f"{m['avg_input_tokens']:<15.1f} {m['num_samples']:<10}")


if __name__ == '__main__':
    main()
