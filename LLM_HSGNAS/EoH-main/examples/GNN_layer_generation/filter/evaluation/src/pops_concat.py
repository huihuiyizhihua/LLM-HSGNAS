import os
import numpy as np
import argparse
import json
import shutil

# =========================================================================
# 1. 路径与参数配置
# =========================================================================
ABS_PATH = os.getcwd()
ROOT_PATH = os.path.abspath(os.path.join(ABS_PATH, "../../.."))

# 输入文件：当前的精英种群 和 本代新生成的种群
conditate_pop_jsonpath = os.path.join(ROOT_PATH, 'result/LLM_generation/conditate_population/conditate_pops.json')
generate_pop_jsonpath = os.path.join(ROOT_PATH, 'result/LLM_generation/generate_pop/generate_pop.json')

parser = argparse.ArgumentParser()
parser.add_argument('--conditate_pop_num', type=int, default=5, help='保留的精英数量')
parser.add_argument('--dataset', type=str, default='Cora', help='当前训练的数据集')
parser.add_argument('--current_round', type=int, default=1, help='当前进化轮次')
parser.add_argument('--total_rounds', type=int, default=20, help='总进化轮次')
args = parser.parse_args()
#import sys; sys.exit(0)   # <-- 加这行（）消融之后要删掉
# =========================================================================
# 2. 加载数据并提取 objective (分数值)
# =========================================================================
conditate_pop_list, generate_pop_list = [], []

# 加载旧精英库
with open(conditate_pop_jsonpath, 'r') as file:
    conditate_data = json.load(file)
for individual in conditate_data:
    val = individual.get('objective', 0)
    conditate_pop_list.append(0 if val is None else val)

# 加载新生成种群
with open(generate_pop_jsonpath, 'r') as file:
    generate_data = json.load(file)
for individual in generate_data:
    val = individual.get('objective', 0)
    generate_pop_list.append(0 if val is None else val)

# =========================================================================
# 3. 排序与优胜劣汰 (基于 Objective 从大到小)
# =========================================================================
seq1 = np.array(conditate_pop_list)
seq2 = np.array(generate_pop_list)

merged_values = np.concatenate((seq1, seq2))
merged_indices = np.concatenate((np.arange(len(seq1)), np.arange(len(seq2))))
merged_sources = np.concatenate((np.full(len(seq1), 'conditate_pop'), np.full(len(seq2), 'generate_pop')))

# 选取分数最高的精英个体
sorted_indices = np.argsort(merged_values)[::-1][:args.conditate_pop_num]

new_condatate_pops = []
for idx in sorted_indices:
    source = merged_sources[idx]
    orig_idx = sorted_original_indices = merged_indices[idx]
    if source == 'conditate_pop':
        new_condatate_pops.append(conditate_data[orig_idx])
    else:
        new_condatate_pops.append(generate_data[orig_idx])

# 过滤掉无效个体
filtered_data = [item for item in new_condatate_pops if item.get("algorithm") and item.get("code")]

# =========================================================================
# 4. 保存结果 (修复了 final_population 未定义的 Bug)
# =========================================================================
# ❗ 关键：将变量名统一为 filtered_data
final_output = filtered_data

# 1. 保存为默认文件 (供下一轮进化循环读取)
default_save_path = conditate_pop_jsonpath
with open(default_save_path, 'w') as f:
    json.dump(final_output, f, indent=4)

# 2. 保存为数据集专属文件 (防止不同数据集实验互相覆盖)
specific_save_path = os.path.join(ROOT_PATH, f"result/LLM_generation/conditate_population/conditate_pops_{args.dataset}.json")
with open(specific_save_path, 'w') as f:
    json.dump(final_output, f, indent=4)

# =========================================================================
# 5. 打印进度
# =========================================================================
print(f"✅ 第 {args.current_round}/{args.total_rounds} 轮种群合并完成！")
print(f"🔥 当前精英库最高分: {max(merged_values):.4f}")
print(f"📁 默认库更新: {default_save_path}")
print(f"📁 专属库备份: {specific_save_path}")