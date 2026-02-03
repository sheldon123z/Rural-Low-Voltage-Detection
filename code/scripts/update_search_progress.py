#!/usr/bin/env python3
"""自动更新 Optuna 搜索进度文件"""
import re
import os
from datetime import datetime

LOG_FILE = "./results/optuna/full_search.log"
PROGRESS_FILE = "./results/optuna/search_progress.md"

def parse_log():
    """解析 Optuna 搜索日志"""
    if not os.path.exists(LOG_FILE):
        print(f"日志文件不存在: {LOG_FILE}")
        return []

    with open(LOG_FILE, 'r') as f:
        content = f.read()

    trials = []

    # 解析每个 Trial 的指标
    metrics_pattern = r'Accuracy: ([\d.]+), Precision: ([\d.]+), Recall: ([\d.]+), F1-score: ([\d.]+)'
    metrics_matches = re.findall(metrics_pattern, content)

    # 解析每个 Trial 的参数
    params_pattern = r'Trial (\d+) finished with value: ([\d.]+) and parameters: ({.*?})\.'
    params_matches = re.findall(params_pattern, content)

    best_f1 = 0.0
    best_trial = -1

    for i, (trial_num, value, params_str) in enumerate(params_matches):
        trial_num = int(trial_num)
        f1 = float(value)

        # 解析参数
        params = {}
        for kv in re.findall(r"'(\w+)': ([\d.e\-]+|'[^']*')", params_str):
            key, val = kv
            try:
                params[key] = float(val) if '.' in val or 'e' in val else int(val)
            except:
                params[key] = val

        # 获取对应的指标
        if i < len(metrics_matches):
            acc, prec, rec, f1_score = metrics_matches[i]
        else:
            acc, prec, rec, f1_score = '?', '?', '?', str(f1)

        is_best = f1 > best_f1
        if is_best:
            best_f1 = f1
            best_trial = trial_num

        trials.append({
            'number': trial_num,
            'params': params,
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1_score,
            'is_best': trial_num == best_trial and f1 == best_f1,
        })

    # 标记最终最佳
    for t in trials:
        t['is_best'] = (t['number'] == best_trial)

    return trials

def generate_markdown(trials):
    """生成 Markdown 进度文件"""
    total_trials = 30
    completed = len(trials)
    best = max(trials, key=lambda t: float(t['f1'])) if trials else None

    md = f"""# VoltageTimesNet_v2 超参数搜索进度

## 搜索配置
- **搜索方法**: Optuna TPE Sampler (multivariate=True)
- **总 Trial 数**: {total_trials}
- **训练 Epochs**: 5
- **数据集**: RuralVoltage/realistic_v2 (16 features)
- **开始时间**: 2026-02-03 10:48:13

## 搜索空间
| 参数 | 范围 |
|------|------|
| d_model | [32, 64, 128, 256] |
| e_layers | [1, 4] |
| d_ff | [32, 64, 128, 256, 512] |
| top_k | [2, 7] |
| num_kernels | [4, 6, 8] |
| lr | [1e-4, 1e-2] (log) |
| batch_size | [32, 64, 128] |
| dropout | [0.0, 0.3] |
| seq_len | [50, 100, 200] |
| anomaly_ratio | [1.5, 5.0] |

## Trial 结果

| Trial | d_model | e_layers | d_ff | top_k | num_kernels | lr | batch | dropout | seq_len | anom_ratio | Acc | Prec | Recall | F1 | Best? |
|:-----:|:-------:|:--------:|:----:|:-----:|:-----------:|:--:|:-----:|:-------:|:-------:|:----------:|:---:|:----:|:------:|:--:|:-----:|
"""

    for t in trials:
        p = t['params']
        mark = "**★**" if t['is_best'] else ""
        bold_s = "**" if t['is_best'] else ""
        bold_e = "**" if t['is_best'] else ""

        md += f"| {bold_s}{t['number']}{bold_e} "
        md += f"| {p.get('d_model', '?')} "
        md += f"| {p.get('e_layers', '?')} "
        md += f"| {p.get('d_ff', '?')} "
        md += f"| {p.get('top_k', '?')} "
        md += f"| {p.get('num_kernels', '?')} "
        md += f"| {float(p.get('lr', 0)):.2e} "
        md += f"| {p.get('batch_size', '?')} "
        md += f"| {float(p.get('dropout', 0)):.3f} "
        md += f"| {p.get('seq_len', '?')} "
        md += f"| {float(p.get('anomaly_ratio', 0)):.2f} "
        md += f"| {t['accuracy']} "
        md += f"| {t['precision']} "
        md += f"| {t['recall']} "
        md += f"| {bold_s}{t['f1']}{bold_e} "
        md += f"| {mark} |\n"

    if best:
        p = best['params']
        md += f"""
## 当前最佳配置 (Trial {best['number']})
```json
{{
  "d_model": {p.get('d_model', '?')},
  "e_layers": {p.get('e_layers', '?')},
  "d_ff": {p.get('d_ff', '?')},
  "top_k": {p.get('top_k', '?')},
  "num_kernels": {p.get('num_kernels', '?')},
  "lr": {p.get('lr', '?')},
  "batch_size": {p.get('batch_size', '?')},
  "dropout": {p.get('dropout', '?')},
  "seq_len": {p.get('seq_len', '?')},
  "anomaly_ratio": {p.get('anomaly_ratio', '?')},
  "f1_score": {best['f1']},
  "recall": {best['recall']},
  "precision": {best['precision']},
  "accuracy": {best['accuracy']}
}}
```
"""

    # 分析趋势
    md += "\n## 关键发现\n\n"

    # 统计 seq_len 分布
    seq_counts = {}
    best_by_seq = {}
    for t in trials:
        sl = t['params'].get('seq_len', '?')
        seq_counts[sl] = seq_counts.get(sl, 0) + 1
        f1 = float(t['f1'])
        if sl not in best_by_seq or f1 > best_by_seq[sl]:
            best_by_seq[sl] = f1

    md += f"- **seq_len 分布**: "
    md += ", ".join([f"{k}={v}次(最佳F1={best_by_seq[k]:.4f})" for k, v in sorted(seq_counts.items())])
    md += "\n"

    # 统计 d_model 分布
    dm_best = {}
    for t in trials:
        dm = t['params'].get('d_model', '?')
        f1 = float(t['f1'])
        if dm not in dm_best or f1 > dm_best[dm]:
            dm_best[dm] = f1
    md += f"- **d_model 最佳F1**: "
    md += ", ".join([f"{k}={v:.4f}" for k, v in sorted(dm_best.items())])
    md += "\n"

    md += f"""
## 与之前最佳对比
| 指标 | 之前最佳 (3 trials) | 当前最佳 (Trial {best['number'] if best else '?'}) | 变化 |
|------|:-------------------:|:------------------:|:----:|
| F1 | 0.7649 | {best['f1'] if best else '?'} | {'+' if best and float(best['f1']) > 0.7649 else ''}{(float(best['f1']) - 0.7649)*100:.1f}% |
| Recall | 0.9631 | {best['recall'] if best else '?'} | {(float(best['recall']) - 0.9631)*100:.1f}% |
| Precision | 0.6343 | {best['precision'] if best else '?'} | {(float(best['precision']) - 0.6343)*100:.1f}% |
| Accuracy | 0.9131 | {best['accuracy'] if best else '?'} | {(float(best['accuracy']) - 0.9131)*100:.1f}% |

## 搜索进度
- **已完成**: {completed}/{total_trials} ({completed*100//total_trials}%)
- **当前状态**: {'✅ 完成' if completed >= total_trials else '🔄 运行中'}
- **最后更新**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

    return md

def main():
    trials = parse_log()
    if not trials:
        print("没有找到已完成的 trial")
        return

    md = generate_markdown(trials)

    with open(PROGRESS_FILE, 'w') as f:
        f.write(md)

    best = max(trials, key=lambda t: float(t['f1']))
    print(f"已更新进度文件: {PROGRESS_FILE}")
    print(f"已完成: {len(trials)}/30 trials")
    print(f"当前最佳: Trial {best['number']}, F1={best['f1']}")

if __name__ == '__main__':
    main()
