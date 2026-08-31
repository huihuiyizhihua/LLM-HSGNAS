#! /bin/bash

# ==========================================
# 实验配置参数
# ==========================================
LLM_generate_num=12
pop_generation_num=20
conditate_pop_num=30

# 🌟DATASETS=("Chameleon" "Squirrel" "Cora" "PubMed" "Photo" "Texas" "Cornell")
DATASET="Cornell"

# ==========================================
# 本地路径配置
# 请将 BASE_DIR 改为你自己的项目根目录（EoH-main 所在目录）
# ==========================================
BASE_DIR="/path/to/your/LLM_HSGNAS/EoH-main"

LLM_generating="$BASE_DIR/examples/GNN_layer_generation/runEoH.py"
training_testing="$BASE_DIR/examples/GNN_layer_generation/filter/evaluation/src/train_model_raw.py"
pops_concat="$BASE_DIR/examples/GNN_layer_generation/filter/evaluation/src/pops_concat.py"

TIMEOUT_DURATION=1800

# 显卡数量设置为 1 (单卡运行)
NUM_GPU=1
PIDS=()

export PYTHONPATH="$BASE_DIR:$PYTHONPATH"

# ==========================================
# 单个模型评估函数
# ==========================================
pyfile_running() {
  local pop_index=$1
  local gpu_id=$2

  # 加上了 --train_rate 0.025 和 --val_rate 0.025
  CUDA_VISIBLE_DEVICES=$gpu_id timeout $TIMEOUT_DURATION python $training_testing --index $pop_index --dataset $DATASET --train_rate 0.025 --val_rate 0.025 --RPMAX  10 &
  PIDS+=($!)
}

# ==========================================
# 主循环流程
# ==========================================
main() {

	# 1. 构建初始种群 (自动根据上面配置的 DATASET 变量选择对应的初始精英文件)
	source_file="$BASE_DIR/examples/result/Initial_Individuals/DenseSplit_${DATASET}_Initial.json"
	destination_file="$BASE_DIR/examples/result/LLM_generation/conditate_population/conditate_pops.json"

    # 如果源文件存在则复制，否则提示错误并退出
    if [ -f "$source_file" ]; then
        cp "$source_file" "$destination_file"
        echo "已成功加载 $DATASET 的初始种群文件"
    else
        echo "错误：找不到初始种群文件 $source_file，请检查路径或数据集名称是否正确！"
        exit 1
    fi

	sum_llm=0
	start_time=$(date +%s.%N)

	for j in $(seq 1 ${pop_generation_num}); do
        echo "=========================================="
        echo "正在运行第 $j 代进化..."
        echo "=========================================="

		# 2. 调用大模型生成新的网络架构
		cd "$BASE_DIR"
		start_time1=$(date +%s.%N)
		python $LLM_generating &
		wait

		end_time1=$(date +%s.%N)
		sum_llm=$(echo "$sum_llm + $end_time1 - $start_time1" | bc)
		echo "大模型生成累计耗时：$sum_llm 秒"

		# 3. 评估生成的网络架构
		cd "$BASE_DIR/examples/GNN_layer_generation/filter/evaluation"

		for i in $(seq 0 ${LLM_generate_num}); do
			gpu_id=$(( (i) % ${NUM_GPU} ))
			echo "正在评估个体编号: ${i} (使用 GPU: ${gpu_id})"
			pyfile_running $i $gpu_id
			if [ $(( (i+1) % ${NUM_GPU} )) -eq 0 ] || [ $i -eq $(($LLM_generate_num - 1)) ]; then
                        	wait "${PIDS[@]}"
				PIDS=()
            fi
	    done
	    wait "${PIDS[@]}"
        # 4. 合并并筛选出本代的精英个体
        python $pops_concat --conditate_pop_num $conditate_pop_num --dataset $DATASET --current_round $j --total_rounds $pop_generation_num &
   wait
	done

	end_time=$(date +%s.%N)
	elapsed_time=$(echo "$end_time - $start_time" | bc)
	echo "大模型生成累计耗时：$sum_llm 秒"
	echo "脚本总运行耗时：$elapsed_time 秒"
}

main
