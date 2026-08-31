import os
import json
import warnings


class GNN_LG():
    def __init__(self) -> None:
        self.debug_mode = False
        from prompts import GetPrompts
        self.prompts = GetPrompts()

        # 定位 generate_pop.json 的路径 (和 train_model_raw.py 保持一致)
        ABS_PATH = os.getcwd()
        ROOT_PATH = os.path.abspath(os.path.join(ABS_PATH, "../../.."))
        self.generation_path = os.path.join(ROOT_PATH, 'result/LLM_generation/generate_pop/generate_pop.json')

    def evaluate(self, code_string):
        """
        因为真实的训练、KV260测速以及分数的保存，
        都已经由 train_model_raw.py 跑完并写进 generate_pop.json 里了，
        所以这里只需要去 JSON 里面把对应的 objective 取出来即可！
        """
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                # 检查 JSON 文件是否存在
                if not os.path.exists(self.generation_path):
                    if self.debug_mode:
                        print(f"Error: JSON file not found at {self.generation_path}")
                    return 0.0

                # 加载 JSON 数据
                with open(self.generation_path, 'r') as file:
                    data = json.load(file)

                # 匹配当前正在评估的代码
                code_string_clean = code_string.strip()

                for individual in data:
                    stored_code = individual.get("code", "").strip()
                    if stored_code == code_string_clean:
                        # 找到了对应的代码，直接返回它的 objective 分数 (即验证集准确率)
                        fitness = float(individual.get("objective", 0.0))
                        return fitness

                # 如果没找到，说明 train_model_raw.py 在端侧跑崩了，或者没生成成功
                if self.debug_mode:
                    print("Warning: Code string not found in JSON. Assigning fitness 0.0")
                return 0.0

        except Exception as e:
            if self.debug_mode:
                print(f"Error in prob.py evaluate: {e}")
            return 0.0