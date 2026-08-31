import os

from eoh.src import eoh_main
from eoh.src.utils.getParas import Paras

from prob import *

# Parameter initilization
paras = Paras()

# Set your local problem
problem_local = GNN_LG()

# Set parameters
paras.set_paras(method="eoh",
                problem=problem_local,

                # LLM 配置（通过环境变量注入，避免密钥泄露）
                #   export LLM_API_ENDPOINT="your-endpoint.com"
                #   export LLM_API_KEY="sk-xxxx"
                #   export LLM_MODEL="gpt-4.1-mini"
                llm_api_endpoint=os.getenv("LLM_API_ENDPOINT", "your-endpoint.com"),
                llm_api_key=os.getenv("LLM_API_KEY", "xxx"),
                llm_model=os.getenv("LLM_MODEL", "gpt-4.1-mini"),

                # ==== 最关键的一行：强制指定种群读取路径，覆盖 getParas.py 中的默认值 ====
                # 通过环境变量注入，避免泄露服务器绝对路径
                #   export LLM_HSGNAS_RESULT_DIR="/path/to/EoH-main/examples/result"
                exp_continue_path=os.path.join(
                    os.getenv("LLM_HSGNAS_RESULT_DIR", "/path/to/EoH-main/examples/result"),
                    "LLM_generation/conditate_population/conditate_pops.json"),
                # =========================================================================

                ec_pop_size=4,
                ec_n_pop=4,
                exp_n_proc=4,
                exp_debug_mode=False,
                eva_numba_decorator=False,
                eva_timeout=60,
                exp_use_continue=True,
                ec_operators=['e1', 'e2', 'dpo'],
                ec_operator_weights=[1, 1, 1],
                )

# initilization
evolution = eoh_main.EVOL(paras)

# run
evolution.run()
