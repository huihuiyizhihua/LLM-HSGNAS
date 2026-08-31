import http.client
import requests
import json
import time  # 新增：用于记录时间戳


class InterfaceAPI:
    def __init__(self, api_endpoint, api_key, model_LLM, debug_mode):
        self.api_endpoint = api_endpoint
        self.api_key = api_key
        self.model_LLM = model_LLM
        self.debug_mode = debug_mode
        self.n_trial = 5

    def get_response(self, prompt_content):
        # print('=====================')
        # print(prompt_content) # "1+1=?"
        payload_explanation = json.dumps(
            {
                "model": self.model_LLM,
                "messages": [
                    # {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt_content}
                ],
                # "temperature": 0.7
            }
        )
        #print('===================下面是提示词内容==========================================================')
        #print(payload_explanation)
        #print('==============================================================================')
        print('=========生成了架构============================================')

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "User-Agent": "Apifox/1.0.0 (https://apifox.com)",
            "Content-Type": "application/json",
            "x-api2d-no-cache": "1",

        }
        # print('=====================')
        # print(headers)

        response = None
        n_trial = 1
        while True:
            n_trial += 1
            if n_trial > self.n_trial:
                return response
            try:
                url = f"https://{self.api_endpoint}/v1/chat/completions"
                res = requests.post(url, headers=headers, data=payload_explanation)

                json_data = res.json()
                response = json_data["choices"][0]["message"]["content"]

                # =========================================================
                # 新增：将 LLM 的请求和回复保存到 JSON 文件中
                # =========================================================
                log_entry = {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "model": self.model_LLM,
                    "prompt": prompt_content,
                    "response": response
                }

                # 以追加模式(a)写入日志文件
                with open("llm_responses_log.json", "a", encoding="utf-8") as log_file:
                    log_file.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
                # =========================================================

                #print('=========llm回复============')
                #print(response)
                #print('=========llm结束              ============')
                break

            except Exception as e:
                if self.debug_mode:
                    print(f"Error in API: {e}. Restarting the process...")
                time.sleep(1)  # 新增：遇到错误时稍微等待1秒，避免短时间内疯狂重试导致被封IP
                continue

        return response