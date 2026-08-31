import re
import time
import os
import json
from ...llm.interface_LLM import InterfaceLLM


class Evolution():

    def __init__(self, api_endpoint, api_key, model_LLM, llm_use_local, llm_local_url, debug_mode, prompts, **kwargs):

        self.prompt_task = prompts.get_task()
        self.prompt_class_name = prompts.get_class_name()
        self.prompt_class_initial_inputs = prompts.get_class_init_inputs()
        self.prompt_forword_func_inputs = prompts.get_forward_func_inputs()
        self.prompt_forword_func_outputs = prompts.get_forward_func_outputs()
        self.prompt_inout_inf = prompts.get_inout_inf()
        self.prompt_other_inf = prompts.get_other_inf()

        if len(self.prompt_class_initial_inputs) > 1:
            self.joined_init_inputs = ", ".join("'" + s + "'" for s in self.prompt_class_initial_inputs)
        else:
            self.joined_init_inputs = "'" + self.prompt_class_initial_inputs[0] + "'"

        if len(self.prompt_forword_func_inputs) > 1:
            self.joined_inputs = ", ".join("'" + s + "'" for s in self.prompt_forword_func_inputs)
        else:
            self.joined_inputs = "'" + self.prompt_forword_func_inputs[0] + "'"

        if len(self.prompt_forword_func_outputs) > 1:
            self.joined_outputs = ", ".join("'" + s + "'" for s in self.prompt_forword_func_outputs)
        else:
            self.joined_outputs = "'" + self.prompt_forword_func_outputs[0] + "'"

        # set LLMs
        self.api_endpoint = api_endpoint
        self.api_key = api_key
        self.model_LLM = model_LLM
        self.debug_mode = debug_mode  # close prompt checking

        self.interface_llm = InterfaceLLM(self.api_endpoint, self.api_key, self.model_LLM, llm_use_local, llm_local_url,
                                          self.debug_mode)

    # ✅ 核心补丁升级版：支持扫描永久日志，防止跨代遗忘
    def _sync_indiv_with_json(self, indiv):
        import glob
        # 结果目录通过环境变量注入，避免泄露服务器绝对路径
        #   export LLM_HSGNAS_RESULT_DIR="/path/to/EoH-main/examples/result"
        result_dir = os.getenv("LLM_HSGNAS_RESULT_DIR", "/path/to/EoH-main/examples/result")
        paths = [
            os.path.join(result_dir, "LLM_generation/generate_pop/generate_pop.json"),
            os.path.join(result_dir, "LLM_generation/conditate_population/conditate_pops.json")
        ]

        # 将所有永久存档的历史结果日志加入检索路径
        log_paths = glob.glob(
            os.path.join(result_dir, "*_test_results_hw_final.json"))
        paths.extend(log_paths)

        code_clean = indiv.get('code', '').strip()
        for path in paths:
            if not os.path.exists(path):
                continue
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                for item in data:
                    if item.get("code", "").strip() == code_clean:
                        # 找回报错信息
                        if "error_msg" in item and item["error_msg"]:
                            indiv["error_msg"] = item["error_msg"]

                        # 找回硬件指标 (兼容两种存放格式)
                        if "hw_metrics_val" in item and item["hw_metrics_val"]:
                            indiv["hw_metrics"] = item["hw_metrics_val"]
                        elif "hw_metrics" in item and item["hw_metrics"]:
                            indiv["hw_metrics"] = item["hw_metrics"]

                        return indiv
            except Exception:
                pass
        return indiv

    # ✅ 封装提取逻辑，统一获取硬件指标及潜在错误
    def _extract_metrics_and_errors(self, indiv):
        metrics_str = ""
        error_str = ""

        if 'error_msg' in indiv and indiv['error_msg']:
            error_str = f"CRITICAL ERROR (Hardware/Execution): {indiv['error_msg']}\n"

        if 'hw_metrics' in indiv:
            m = indiv['hw_metrics']
            if 'error' in m and m['error']:
                error_str = f"CRITICAL ERROR (Deployment Failed): {m['error']}\n"

            metrics_str = f"Hardware Metrics Feedback (Primary KV260): ACC={m.get('acc', 0):.4f}, LATENCY={m.get('lat', 999):.6f}s\n"
            if 'devices' in m:
                devs = m['devices']
                metrics_str += "Detailed Multi-Device Performance:\n"
                for dev_name in ["CPU", "GPU", "KV260", "Jetson"]:
                    if dev_name in devs:
                        d = devs[dev_name]

                        metrics_str += f" - {dev_name}: ACC={d.get('acc', 0) * 100:.2f}%, LAT={d.get('lat', 999):.6f}s, THROUGHPUT={d.get('thr', 0):.2f}, ENERGY={d.get('energy', 0):.4f}J\n"
        return metrics_str, error_str

    def get_prompt_i1(self):
        # 修正：将原代码中未定义的 self.prompt_func_name 统一为 self.prompt_class_name
        prompt_content = self.prompt_task + "\n" \
                                            "First, describe your new algorithm and main steps in one sentence. \
                                          The description must be inside a brace. Next, implement it in Python as a class named \
                                          " + self.prompt_class_name + ". This class initial should accept " + str(
            len(self.prompt_class_initial_inputs)) + " input(s): " \
                         + self.joined_init_inputs + ". The forward function should accept " + str(
            len(self.prompt_forword_func_inputs)) + " input(s): " \
                         + self.joined_inputs + ". The function should return " + str(
            len(self.prompt_forword_func_outputs)) + " output(s): " \
                         + self.joined_outputs + ". " + self.prompt_inout_inf + " " \
                         + self.prompt_other_inf + "\n" + "Prohibit the use of from xxx import *." + "Please add the end identifier '#code_end' at the end of the code segment" + "Do not give additional explanations."
        return prompt_content

    def get_prompt_e1(self, indivs):
        prompt_indiv = ""
        for i in range(len(indivs)):
            # 实时同步 JSON 数据
            indiv = self._sync_indiv_with_json(indivs[i])
            metrics_str, error_str = self._extract_metrics_and_errors(indiv)

            prompt_indiv += "No." + str(i + 1) + " algorithm and the corresponding code are: \n" + indiv[
                'algorithm'] + "\n" + indiv['code'] + "\nScore:\n" + str(
                indiv['objective']) + "\n" + metrics_str + error_str + "\n"

        prompt_content = self.prompt_task + "\n" \
                                            "I have " + str(
            len(indivs)) + " existing spectral graph neural network layers with their codes as follows: \n" \
                         + prompt_indiv + \
                         "Please help me create a new spectral graph neural network layers that has a totally different form from the given ones. \n" \
                         "Please carefully analyze the 'Hardware Metrics Feedback'. Focus on optimizing latency and throughput for the edge devices without losing accuracy.\n" \
                         "IMPORTANT: If any provided algorithm contains a 'CRITICAL ERROR' (e.g., ONNX tracing failure, unsupported operators), you MUST carefully analyze the error reason and ensure your new code strictly avoids the same mistake.\n" \
                         "First, describe your new network layers and main steps in one sentence. \
                         Your new algorithm description must be inside a brace. Next, implement it in Python as a class named \
                         " + self.prompt_class_name + ". This class initial should accept " + str(
            len(self.prompt_class_initial_inputs)) + " input(s): " \
                         + self.joined_init_inputs + ". The forward function should accept " + str(
            len(self.prompt_forword_func_inputs)) + " input(s): " + self.joined_inputs + ",and return " + str(
            len(self.prompt_forword_func_outputs)) + " output(s): " \
                         + self.joined_outputs + ". " + self.prompt_inout_inf + " " \
                         + self.prompt_other_inf + "\n" + "Please add the end identifier '#code_end' at the end of the code segment" + "Do not give additional explanations."
        return prompt_content

    def get_prompt_e2(self, indivs):
        prompt_indiv = ""
        for i in range(len(indivs)):
            # 实时同步 JSON 数据
            indiv = self._sync_indiv_with_json(indivs[i])
            metrics_str, error_str = self._extract_metrics_and_errors(indiv)

            prompt_indiv += "No." + str(i + 1) + " algorithm and the corresponding code are: \n" + indiv[
                'algorithm'] + "\n" + indiv['code'] + "\nScore:\n" + str(
                indiv['objective']) + "\n" + metrics_str + error_str + "\n"

        prompt_content = self.prompt_task + "\n" \
                                            "I have " + str(
            len(indivs)) + " existing spectral graph neural network layers with their codes as follows: \n" \
                         + prompt_indiv + \
                         "Please help me create a new spectral graph neural network layers that has a totally different form from the given ones but can be motivated from them. \n" \
                         "Please carefully analyze the 'Hardware Metrics Feedback' to balance accuracy and efficiency across all devices.\n" \
                         "IMPORTANT: If any provided algorithm contains a 'CRITICAL ERROR', you MUST ensure your new architecture is hardware-compatible and bypasses these issues.\n" \
                         "Firstly, identify the common backbone idea in the provided algorithms. Secondly, based on the backbone idea describe your new algorithm in one sentence. \
                         Your new algorithm description must be inside a brace. Thirdly, implement it in Python as a class named \
                         " + self.prompt_class_name + ". This class initial should accept " + str(
            len(self.prompt_class_initial_inputs)) + " input(s): " \
                         + self.joined_init_inputs + ". The forward function should accept " + str(
            len(self.prompt_forword_func_inputs)) + " input(s): " + self.joined_inputs + ",and return " + str(
            len(self.prompt_forword_func_outputs)) + " output(s): " \
                         + self.joined_outputs + ". " + self.prompt_inout_inf + " " \
                         + self.prompt_other_inf + "\n" + "Please add the end identifier '#code_end' at the end of the code segment" + "Do not give additional explanations."
        return prompt_content

    def get_prompt_dpo(self, indiv1, indiv2):
        # 实时同步 JSON 数据
        indiv1 = self._sync_indiv_with_json(indiv1)
        indiv2 = self._sync_indiv_with_json(indiv2)

        metrics_str1, error_str1 = self._extract_metrics_and_errors(indiv1)
        metrics_str2, error_str2 = self._extract_metrics_and_errors(indiv2)

        prompt_content = self.prompt_task + "\n" \
                                            "I have two spectral graph neural network layers with their code and hardware performance as follows. \n\n\
                                          Algorithm description 1: " + indiv1.get('algorithm', '') + "\n\
Code 1:\n\
" + indiv1['code'] + "\n\
Score 1:\n\
" + str(indiv1['objective']) + "\n\
" + metrics_str1 + error_str1 + "\n\
Algorithm description 2: " + indiv2.get('algorithm', '') + "\n\
Code 2:\n\
" + indiv2['code'] + "\n\
Score 2:\n\
" + str(indiv2['objective']) + "\n\
" + metrics_str2 + error_str2 + "\n\
First, analyze the principles of both; \n Second, by comparing their scores, Hardware Metrics, and any CRITICAL ERRORS, evaluate why the filter with the better overall performance (or the one that successfully bypassed deployment errors) is more suitable for the graphs and edge devices. \n" \
                                "Finally, help me to create a new spectral graph neural network layer, and describe your new algorithm and main steps in one sentence.\
                            Your new algorithm description must be inside a brace. Next, implement it in Python as a class named \
                            " + self.prompt_class_name + ". This class initial should accept " + str(
            len(self.prompt_class_initial_inputs)) + " input(s): " \
                         + self.joined_init_inputs + ". The forward function should accept " + str(
            len(self.prompt_forword_func_inputs)) + " input(s): " + self.joined_inputs + ",and return " + str(
            len(self.prompt_forword_func_outputs)) + " output(s): " \
                         + self.joined_outputs + ". " + self.prompt_inout_inf + " " \
                         + self.prompt_other_inf + "\n" + "Please add the end identifier '#code_end' at the end of the code segment" + "Do not give additional explanations."
        return prompt_content

    def _get_alg(self, prompt_content):
        response = self.interface_llm.get_response(prompt_content)
        algorithm = re.findall(r"\{(.*)\}", response, re.DOTALL)
        if len(algorithm) == 0:
            if 'python' in response:
                algorithm = re.findall(r'^.*?(?=python)', response, re.DOTALL)
            elif 'import' in response:
                algorithm = re.findall(r'^.*?(?=import)', response, re.DOTALL)
            else:
                algorithm = re.findall(r'^.*?(?=class)', response, re.DOTALL)

        code = re.findall(r"import .*#code_end", response, re.DOTALL)
        if len(code) == 0:
            code = re.findall(r"import .*#code_end", response, re.DOTALL)

        n_retry = 1
        while (len(algorithm) == 0 or len(code) == 0):
            if self.debug_mode:
                print("Error: algorithm or code not identified, wait 1 seconds and retrying ... ")

            response = self.interface_llm.get_response(prompt_content)

            algorithm = re.findall(r"\{(.*)\}", response, re.DOTALL)
            if len(algorithm) == 0:
                if 'python' in response:
                    algorithm = re.findall(r'^.*?(?=python)', response, re.DOTALL)
                elif 'import' in response:
                    algorithm = re.findall(r'^.*?(?=import)', response, re.DOTALL)
                else:
                    algorithm = re.findall(r'^.*?(?=class)', response, re.DOTALL)

            code = re.findall(r"import .*#code_end", response, re.DOTALL)
            if len(code) == 0:
                code = re.findall(r"import .*#code_end", response, re.DOTALL)

            if n_retry > 3:
                break
            n_retry += 1

        #algorithm = algorithm[0]
        # 这样可以同时去掉两端的空格和可能残余的大括号
        algorithm = algorithm[0].strip().strip('{}').strip()
        code = code[0]

        return [code, algorithm]

    def i1(self):

        prompt_content = self.get_prompt_i1()

        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ i1 ] : \n", prompt_content)
            print(">>> Press 'Enter' to continue")
            input()

        [code_all, algorithm] = self._get_alg(prompt_content)

        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            print(">>> Press 'Enter' to continue")
            input()

        return [code_all, algorithm]

    def e1(self, parents):

        prompt_content = self.get_prompt_e1(parents)

        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ e1 ] : \n", prompt_content)
            print(">>> Press 'Enter' to continue")
            input()

        [code_all, algorithm] = self._get_alg(prompt_content)

        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            print(">>> Press 'Enter' to continue")
            input()

        return [code_all, algorithm]

    def e2(self, parents):

        prompt_content = self.get_prompt_e2(parents)

        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ e2 ] : \n", prompt_content)
            print(">>> Press 'Enter' to continue")
            input()

        [code_all, algorithm] = self._get_alg(prompt_content)

        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            print(">>> Press 'Enter' to continue")
            input()

        return [code_all, algorithm]

    def dpo(self, parents1, parents2):

        prompt_content = self.get_prompt_dpo(parents1, parents2)

        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ dpo ] : \n", prompt_content)
            print(">>> Press 'Enter' to continue")
            input()

        [code_all, algorithm] = self._get_alg(prompt_content)

        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            print(">>> Press 'Enter' to continue")
            input()

        return [code_all, algorithm]