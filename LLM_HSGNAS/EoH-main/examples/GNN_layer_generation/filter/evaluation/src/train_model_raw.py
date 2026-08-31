#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Distributed under terms of the MIT license.
import time
import os
import argparse
import sys
import json
import types
import warnings
import numpy as np
import random
import csv
import subprocess
import fcntl  # 引入文件锁模块，用于解决多进程并发写入覆盖问题
import glob
import threading
import torch
import torch.nn.functional as F
import copy  # 用于深度拷贝模型最优权重
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch_geometric.loader import DataLoader as DATALOADER

# 导入项目依赖 (请确保这些文件与此脚本在同级或可引用的 PYTHONPATH 下)
from dataset_utils import *
from utils import random_planetoid_splits, random_graph_splits
from GNN_models import *

# =========================================================================
# 🚦 拦截器设置：强制淘汰不兼容硬件的代码
# =========================================================================
# warnings.filterwarnings("error", category=UserWarning, message=".*aten::index.*Byte.*")
# warnings.filterwarnings("error", category=torch.jit.TracerWarning,
#                         message=".*Converting a tensor to a Python boolean.*")
# warnings.filterwarnings("error", category=torch.jit.TracerWarning, message=".*Iterating over a tensor.*")

ABS_PATH = os.getcwd()
ROOT_PATH = os.path.abspath(os.path.join(ABS_PATH, "../../.."))
Generation_Path = os.path.join(ROOT_PATH, 'result/LLM_generation/generate_pop/generate_pop.json')

# =========================================================================
# 🌐 远程硬件配置信息 (引入环境变量 fallback，增强健壮性)
# =========================================================================
from edge_measurer import (
    JETSON_IP,
    JETSON_USER,
    KV260_IP,
    KV260_USER,
    evaluate_remote_hardware,
)

# =========================================================================
# 本机能耗采样工具 (对齐 NAS_Bench_Graph)
# =========================================================================
RAPL_ENERGY_FILE = None


def _init_rapl():
    global RAPL_ENERGY_FILE
    if RAPL_ENERGY_FILE is not None: return
    candidates = glob.glob("/sys/class/powercap/intel-rapl:*/energy_uj")
    if candidates:
        RAPL_ENERGY_FILE = candidates[0]


def _read_rapl_uj():
    """
    对齐 nas_bench 逻辑：读取时直接转换为 float 以保留原始精度
    """
    rapl_path = "/sys/class/powercap/intel-rapl:0/energy_uj"
    if os.path.exists(rapl_path):
        try:
            with open(rapl_path, "r") as f:
                # 关键点：立即转为 float
                return float(f.read().strip())
        except:
            pass
    return -1.0


class GpuPowerSampler:
    def __init__(self, interval_s=0.02, gpu_index=0):
        self.interval_s = interval_s
        self.gpu_index = gpu_index
        self._stop = threading.Event()
        self.samples_w = []
        self._thread = None
        self._mode = None

    def start(self):
        self.samples_w = []
        self._stop.clear()
        try:
            import pynvml
            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
            self._mode = "nvml"
        except:
            self._mode = "smi"

        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while not self._stop.is_set():
            p = None
            if self._mode == "nvml":
                try:
                    import pynvml
                    p = pynvml.nvmlDeviceGetPowerUsage(self._handle) / 1000.0
                except:
                    pass
            elif self._mode == "smi":
                try:
                    out = subprocess.check_output(
                        ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits",
                         f"--id={self.gpu_index}"],
                        stderr=subprocess.DEVNULL, text=True).strip()
                    p = float(out.splitlines()[0])
                except:
                    pass
            if p is not None: self.samples_w.append(float(p))
            time.sleep(self.interval_s)

    def stop(self):
        self._stop.set()
        if self._thread: self._thread.join(timeout=1.0)

    def avg_power_w(self):
        return sum(self.samples_w) / len(self.samples_w) if self.samples_w else None


# =========================================================================
# 1. 辅助函数：数据集切分与动态代码加载
# =========================================================================

def get_all_split_idx(dataset):
    root_idx_dir = './data/TUs/'
    if not os.path.exists(root_idx_dir):
        os.makedirs(root_idx_dir)
    all_idx = {}
    if not (os.path.exists(os.path.join(root_idx_dir, dataset.name + '_train.index'))):
        print("[!] Splitting the data into train/val/test ...")
        k_splits = 10
        cross_val_fold = StratifiedKFold(n_splits=k_splits, shuffle=True)
        graph_labels = [data.y.item() for data in dataset]
        for indexes in cross_val_fold.split(range(len(dataset)), graph_labels):
            remain_index, test_index = indexes[0], indexes[1]
            remain_labels = [graph_labels[i] for i in remain_index]
            train_idx, val_idx = train_test_split(remain_index, test_size=0.111, stratify=remain_labels)
            with open(os.path.join(root_idx_dir, dataset.name + '_train.index'), 'a+') as f_train, \
                    open(os.path.join(root_idx_dir, dataset.name + '_val.index'), 'a+') as f_val, \
                    open(os.path.join(root_idx_dir, dataset.name + '_test.index'), 'a+') as f_test:
                csv.writer(f_train).writerow(list(train_idx))
                csv.writer(f_val).writerow(list(val_idx))
                csv.writer(f_test).writerow(list(test_index))
        print("[!] Splitting done!")
    for section in ['train', 'val', 'test']:
        with open(os.path.join(root_idx_dir, dataset.name + '_' + section + '.index'), 'r') as f:
            reader = csv.reader(f)
            all_idx[section] = [list(map(int, idx)) for idx in reader]
    return all_idx


def reading_code(load_pop_path, index=0):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with open(load_pop_path, 'r') as file:
                content = file.read().strip()
                data = json.loads(content) if content else []
            if len(data) <= index: return None
            Model_layer_code = data[index]
            Layer_module = types.ModuleType("Layer_module")
            exec(Model_layer_code["code"], Layer_module.__dict__)
            sys.modules[Layer_module.__name__] = Layer_module
            return Layer_module.GNN_Layer
    except Exception as e:
        print(f"Error-CodeLoading: {str(e)}")
        return None


# =========================================================================
# 2. 核心训练函数 (已修改：返回验证集最优模型权重)
# =========================================================================

def RunExp_GC(args, dataset, data, Net, percls_trn, val_lb, Layer_module, train_loader, val_loader, test_loader):
    def train(model, optimizer, loader, device):
        model.train()
        total_loss = 0
        for data in loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.batch)
            loss = F.nll_loss(out, data.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    def test(model, loader, device):
        model.eval()
        accs, losses = [], []
        with torch.no_grad():
            for data in loader:
                data = data.to(device)
                out = model(data.x, data.edge_index, data.batch)
                pred = out.max(1)[1]
                acc = pred.eq(data.y).sum().item() / data.num_graphs
                loss = F.nll_loss(out, data.y)
                accs.append(acc)
                losses.append(loss.item())
        return sum(accs) / len(accs), sum(losses) / len(losses)

    appnp_net = Net(dataset.num_features, dataset.num_classes, Layer_module)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = appnp_net.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_acc, best_val_loss, test_acc = 0, float('inf'), 0
    val_loss_history = []
    run_train_time, run_test_time = 0.0, 0.0

    # 初始化最优模型权重快照
    best_model_weights = copy.deepcopy(model.state_dict())

    for epoch in range(args.epochs):
        t0 = time.time()
        train_loss = train(model, optimizer, train_loader, device)
        run_train_time += (time.time() - t0)

        t1 = time.time()
        val_acc, val_loss = test(model, val_loader, device)
        tmp_test_acc, _ = test(model, test_loader, device)
        run_test_time += (time.time() - t1)

        if val_loss < best_val_loss:
            best_val_acc, best_val_loss, test_acc = val_acc, val_loss, tmp_test_acc
            # 当验证集最优时，更新权重快照
            best_model_weights = copy.deepcopy(model.state_dict())

        val_loss_history.append(val_loss)
        if args.early_stopping > 0 and epoch > args.early_stopping:
            if val_loss > torch.tensor(val_loss_history[-(args.early_stopping + 1):-1]).mean().item(): break

    # 在返回模型供硬件评估前，加载最优权重
    model.load_state_dict(best_model_weights)
    return test_acc, best_val_acc, best_val_loss, train_loss, run_train_time, run_test_time, model, data


def RunExp(args, dataset, data, Net, percls_trn, val_lb, Layer_module):
    def train(model, optimizer, data, dprate):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)[data.train_mask]
        loss = F.nll_loss(out, data.y[data.train_mask])
        loss.backward()
        optimizer.step()

    def test(model, data):
        model.eval()
        with torch.no_grad(): logits = model(data.x, data.edge_index)
        accs, losses = [], []
        for _, mask in data('train_mask', 'val_mask', 'test_mask'):
            pred = logits[mask].max(1)[1]
            acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
            loss = F.nll_loss(logits[mask], data.y[mask])
            accs.append(acc)
            losses.append(loss.item())
        return accs, losses

    appnp_net = Net(dataset.num_features, dataset.num_classes, Layer_module)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = random_planetoid_splits(data, dataset.num_classes, percls_trn, val_lb)
    model, data = appnp_net.to(device), data.to(device)

    if args.net in ['APPNP', 'GPRGNN']:
        optimizer = torch.optim.Adam([
            {'params': model.lin1.parameters(), 'weight_decay': args.weight_decay, 'lr': args.lr},
            {'params': model.lin2.parameters(), 'weight_decay': args.weight_decay, 'lr': args.lr},
            {'params': model.prop1.parameters(), 'weight_decay': 0.0, 'lr': args.lr}
        ], lr=args.lr)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_acc, test_acc, best_val_loss = 0, 0, float('inf')
    val_loss_history = []
    run_train_time, run_test_time = 0.0, 0.0

    # 初始化最优模型权重快照
    best_model_weights = copy.deepcopy(model.state_dict())

    for epoch in range(args.epochs):
        t0 = time.time()
        train(model, optimizer, data, args.dprate)
        run_train_time += (time.time() - t0)

        t1 = time.time()
        [train_acc, val_acc, tmp_test_acc], [train_loss, val_loss, tmp_test_loss] = test(model, data)
        run_test_time += (time.time() - t1)

        if val_loss < best_val_loss:
            best_val_acc, best_val_loss, test_acc = val_acc, val_loss, tmp_test_acc
            # 当验证集最优时，更新权重快照
            best_model_weights = copy.deepcopy(model.state_dict())

        val_loss_history.append(val_loss)
        if args.early_stopping > 0 and epoch > args.early_stopping:
            if val_loss > torch.tensor(val_loss_history[-(args.early_stopping + 1):-1]).mean().item(): break

    # 在返回模型供硬件评估前，加载最优权重
    model.load_state_dict(best_model_weights)
    return test_acc, best_val_acc, best_val_loss, train_loss, run_train_time, run_test_time, model, data


# =========================================================================
# 2.5 Local hardware evaluation module: CPU and GPU (supports validation/test masks)
# =========================================================================

def evaluate_local_hardware(model, data, device_type='gpu', runs=50, mask_type='val_mask'):
    """
    高精度 CPU/GPU 能耗评估 (对齐 NAS_Bench_Graph 逻辑)
    """
    device = torch.device('cuda' if device_type == 'gpu' and torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    y = data.y.to(device)
    num_nodes = int(x.size(0))

    if hasattr(data, mask_type):
        m = getattr(data, mask_type).to(device)
    else:
        m = data.test_mask.to(device)

    # 预热处理
    with torch.no_grad():
        for _ in range(10): _ = model(x, edge_index)
    if device.type == 'cuda': torch.cuda.synchronize()

    # ====== 启动能耗采样 ======
    e0_uj = None
    gpu_sampler = None
    if device_type == 'cpu':
        _init_rapl()
        e0_uj = _read_rapl_uj()  # 获取 float 类型的起始能量
    elif device_type == 'gpu':
        gpu_sampler = GpuPowerSampler(interval_s=0.02, gpu_index=0)
        gpu_sampler.start()

    t0 = time.time()
    with torch.no_grad():
        for _ in range(runs):
            out = model(x, edge_index)
    if device.type == 'cuda': torch.cuda.synchronize()
    t1 = time.time()
    total_time_s = t1 - t0

    # ====== 高精度能耗结算 ======
    energy_per_infer = 0.0
    if device_type == 'cpu':
        e1_uj = _read_rapl_uj()  # 获取 float 类型的结束能量
        if e0_uj >= 0 and e1_uj >= 0:
            delta_uj = e1_uj - e0_uj
            if delta_uj < 0:  # 处理器 32 位寄存器溢出情况
                delta_uj += 2 ** 32
            # 核心公式：((e1 - e0) / 1e6) / runs
            energy_per_infer = (delta_uj / 1e6) / runs
        else:
            # RAPL 不可用时回退为 0.0，不造假数据
            energy_per_infer = 0.0

    elif device_type == 'gpu' and gpu_sampler is not None:
        gpu_sampler.stop()
        avg_p_w = gpu_sampler.avg_power_w()
        if avg_p_w is not None:
            energy_per_infer = (avg_p_w * total_time_s) / runs

    pred = out.max(1)[1]
    acc = pred[m].eq(y[m]).sum().item() / m.sum().item() if m.sum().item() > 0 else 0
    avg_lat = total_time_s / runs
    thr = num_nodes / avg_lat if avg_lat > 0 else 0.0

    return acc, avg_lat, thr, energy_per_infer




# =========================================================================
# 3. 主流程
# =========================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=0.001)
    parser.add_argument('--early_stopping', type=int, default=200)
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--train_rate', type=float, default=0.6)
    parser.add_argument('--val_rate', type=float, default=0.2)
    parser.add_argument('--K', type=int, default=10)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--dprate', type=float, default=0.5)
    parser.add_argument('--C', type=int)
    parser.add_argument('--Init', type=str, choices=['SGC', 'PPR', 'NPPR', 'Random', 'WS', 'Null'], default='PPR')
    parser.add_argument('--Gamma', default=None)
    parser.add_argument('--ppnp', default='GPR_prop', choices=['PPNP', 'GPR_prop'])
    parser.add_argument('--heads', default=8, type=int)
    parser.add_argument('--output_heads', default=1, type=int)
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--Task', default='NC')
    parser.add_argument('--dataset', default='PubMed')
    parser.add_argument('--batch_size', default=300)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--RPMAX', type=int, default=3)
    parser.add_argument('--net', type=str, default='LLM_FILTER')
    parser.add_argument('--cpu_runs', type=int, default=3, help='CPU 推理重复次数')
    parser.add_argument('--gpu_runs', type=int, default=30, help='GPU 推理重复次数（需较多以稳定功率采样）')
    parser.add_argument('--kv260_runs', type=int, default=1, help='KV260 推理重复次数')
    parser.add_argument('--jetson_runs', type=int, default=3, help='Jetson 推理重复次数')
    args = parser.parse_args()

    JSON_file = Generation_Path

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    Net = LLM_FILTER

    # ✅ 索引越界保护
    try:
        with open(JSON_file, 'r') as _f:
            _pre_data = json.loads(_f.read().strip() or '[]')
        if args.index >= len(_pre_data):
            print(f"⏭️ 索引 {args.index} 超出范围 (文件仅 {len(_pre_data)} 条)，跳过。")
            sys.exit(0)
    except Exception:
        pass

    Layer_module = reading_code(JSON_file, index=args.index)

    if Layer_module is None:
        print(f"❌ 架构 {args.index} 存在严重语法错误，自动赋予 0 分。")
        error_msg = "Syntax/Compilation Error: 架构代码存在语法错误或非法导入。"
        code_str = ""
        try:
            with open(JSON_file, 'r+') as f:
                fcntl.flock(f, fcntl.LOCK_EX)
                content = f.read().strip()
                data_list = json.loads(content) if content else []
                if len(data_list) > args.index:
                    code_str = data_list[args.index].get("code", "")
                    data_list[args.index]["objective"] = 0.0
                    data_list[args.index]["error_msg"] = error_msg
                    data_list[args.index]["hw_metrics"] = {
                        "acc": 0.0, "lat": 999.0, "lat_std": 0.0, "thr": 0.0, "energy": 999.0,
                        "devices": {
                            "CPU": {"acc": 0.0, "lat": 999.0, "thr": 0.0, "energy": 999.0},
                            "GPU": {"acc": 0.0, "lat": 999.0, "thr": 0.0, "energy": 999.0},
                            "KV260": {"acc": 0.0, "lat": 999.0, "thr": 0.0, "energy": 999.0},
                            "Jetson": {"acc": 0.0, "lat": 999.0, "thr": 0.0, "energy": 999.0}
                        }
                    }
                    f.seek(0)
                    f.truncate()
                    json.dump(data_list, f, indent=4, ensure_ascii=False)
                fcntl.flock(f, fcntl.LOCK_UN)
        except Exception:
            pass

        # ✅ 新增：把语法错误永久存档
        LOG_PATH = os.path.join(ROOT_PATH, "result", f"{args.dataset}_test_results_hw_final.json")
        record = {
            "index": args.index, "dataset": args.dataset,
            "val_acc_mean": 0.0, "test_acc_mean": 0.0, "test_acc_std": 0.0,
            "error_msg": error_msg,
            "hw_metrics_val": {}, "hw_metrics_test": {}, "code": code_str
        }
        if not os.path.exists(LOG_PATH):
            with open(LOG_PATH, 'w') as f: json.dump([], f)
        with open(LOG_PATH, 'r+') as ext_file:
            fcntl.flock(ext_file, fcntl.LOCK_EX)
            try:
                extra_data = json.loads(ext_file.read().strip())
            except json.JSONDecodeError:
                extra_data = []
            extra_data.append(record)
            ext_file.seek(0)
            ext_file.truncate()
            json.dump(extra_data, ext_file, indent=4, ensure_ascii=False)
            fcntl.flock(ext_file, fcntl.LOCK_UN)

        sys.exit(0)

    try:
        Results0 = []
        # ✅ 收集边缘设备部署失败信息（CPU/GPU 正常时保留报错供下一轮 LLM 恢复）
        edge_fail_info = {"KV260": "", "Jetson": ""}

        # 记录评估集 (val_mask) 测试结果
        Results_CPU_val, Results_GPU_val, Results_KV260_val, Results_Jetson_val = [], [], [], []
        # 记录测试集 (test_mask) 测试结果
        Results_CPU_test, Results_GPU_test, Results_KV260_test, Results_Jetson_test = [], [], [], []

        total_train_time = 0.0
        total_test_time = 0.0

        if args.Task == 'NC':
            dname = args.dataset
            dataset, data = DataLoader(dname)
            RPMAX = args.RPMAX
            percls_trn = int(round(args.train_rate * len(data.y) / dataset.num_classes))
            val_lb = int(round(args.val_rate * len(data.y)))
            args.C = len(data.y.unique())

            print(f"\n🚀 开始架构 {args.index} 的 4 端软硬件评估 (RPMAX={RPMAX}, 包含 Val 和 Test 独立测速)...")
            for RP in tqdm(range(RPMAX)):
                # 第一步：GPU 训练，返回的是最优验证集的快照
                test_acc, best_val_acc, best_val_loss, train_loss, tr_time, te_time, model, data_trained = RunExp(
                    args, dataset, data, Net, percls_trn, val_lb, Layer_module)
                Results0.append([test_acc, best_val_acc])
                total_train_time += tr_time
                total_test_time += te_time

                # 第二步：针对 评估集(Val) 的四设备测速
                Results_CPU_val.append(
                    evaluate_local_hardware(model, data_trained, device_type='cpu', runs=args.cpu_runs, mask_type='val_mask'))
                if torch.cuda.is_available():
                    Results_GPU_val.append(
                        evaluate_local_hardware(model, data_trained, device_type='gpu', runs=args.gpu_runs, mask_type='val_mask'))
                try:
                    Results_KV260_val.append(
                        evaluate_remote_hardware(model, data_trained, dname, args.index, KV260_IP, KV260_USER, "kv260",
                                                 runs=args.kv260_runs, mask_type='val_mask'))
                except Exception as e:
                    print(f"[Warn] KV260 (Val) Error: {e}");
                    edge_fail_info["KV260"] = (edge_fail_info["KV260"] + f" [val]{e}").strip()
                    Results_KV260_val.append((0.0, 999.0, 0.0, 0.0))
                try:
                    Results_Jetson_val.append(
                        evaluate_remote_hardware(model, data_trained, dname, args.index, JETSON_IP, JETSON_USER,
                                                 "jetson", runs=args.jetson_runs, mask_type='val_mask'))
                except Exception as e:
                    print(f"[Warn] Jetson (Val) Error: {e}");
                    edge_fail_info["Jetson"] = (edge_fail_info["Jetson"] + f" [val]{e}").strip()
                    Results_Jetson_val.append((0.0, 999.0, 0.0, 0.0))

                # 第三步：针对 测试集(Test) 的四设备测速
                Results_CPU_test.append(
                    evaluate_local_hardware(model, data_trained, device_type='cpu', runs=args.cpu_runs, mask_type='test_mask'))
                if torch.cuda.is_available():
                    Results_GPU_test.append(
                        evaluate_local_hardware(model, data_trained, device_type='gpu', runs=args.gpu_runs, mask_type='test_mask'))
                try:
                    Results_KV260_test.append(
                        evaluate_remote_hardware(model, data_trained, dname, args.index, KV260_IP, KV260_USER, "kv260",
                                                 runs=args.kv260_runs, mask_type='test_mask'))
                except Exception as e:
                    print(f"[Warn] KV260 (Test) Error: {e}");
                    edge_fail_info["KV260"] = (edge_fail_info["KV260"] + f" [test]{e}").strip()
                    Results_KV260_test.append((0.0, 999.0, 0.0, 0.0))
                try:
                    Results_Jetson_test.append(
                        evaluate_remote_hardware(model, data_trained, dname, args.index, JETSON_IP, JETSON_USER,
                                                 "jetson", runs=args.jetson_runs, mask_type='test_mask'))
                except Exception as e:
                    print(f"[Warn] Jetson (Test) Error: {e}");
                    edge_fail_info["Jetson"] = (edge_fail_info["Jetson"] + f" [test]{e}").strip()
                    Results_Jetson_test.append((0.0, 999.0, 0.0, 0.0))

        elif args.Task == 'GC':
            dname = args.dataset
            dataset, data = DataLoader_GC(dname)
            RPMAX = args.RPMAX
            percls_trn = int(round(args.train_rate * len(data.y) / dataset.num_classes))
            val_lb = int(round(args.val_rate * len(data.y)))
            args.C = len(data.y.unique())
            all_idx = get_all_split_idx(dataset)

            print(f"\n🚀 开始架构 {args.index} 的 4 端软硬件同步评估 (RPMAX={RPMAX}, 包含 Val 和 Test 独立测速)...")
            for RP in tqdm(range(RPMAX)):
                train_loader = DATALOADER([dataset[i] for i in all_idx['train'][RP]], batch_size=args.batch_size,
                                          shuffle=True)
                val_loader = DATALOADER([dataset[i] for i in all_idx['val'][RP]], batch_size=args.batch_size,
                                        shuffle=False)
                test_loader = DATALOADER([dataset[i] for i in all_idx['test'][RP]], batch_size=args.batch_size,
                                         shuffle=False)

                test_acc, best_val_acc, best_val_loss, train_loss, tr_time, te_time, model, data_trained = RunExp_GC(
                    args, dataset, data, Net, percls_trn, val_lb, Layer_module, train_loader, val_loader, test_loader)
                Results0.append([test_acc, best_val_acc])
                total_train_time += tr_time
                total_test_time += te_time

                Results_CPU_val.append(
                    evaluate_local_hardware(model, data_trained, device_type='cpu', runs=args.cpu_runs, mask_type='val_mask'))
                if torch.cuda.is_available():
                    Results_GPU_val.append(
                        evaluate_local_hardware(model, data_trained, device_type='gpu', runs=args.gpu_runs, mask_type='val_mask'))
                try:
                    Results_KV260_val.append(
                        evaluate_remote_hardware(model, data_trained, dname, args.index, KV260_IP, KV260_USER, "kv260",
                                                 runs=args.kv260_runs, mask_type='val_mask'))
                except Exception as e:
                    print(f"[Warn] KV260 (Val) Error: {e}");
                    edge_fail_info["KV260"] = (edge_fail_info["KV260"] + f" [val]{e}").strip()
                    Results_KV260_val.append((0.0, 999.0, 0.0, 0.0))
                try:
                    Results_Jetson_val.append(
                        evaluate_remote_hardware(model, data_trained, dname, args.index, JETSON_IP, JETSON_USER,
                                                 "jetson", runs=args.jetson_runs, mask_type='val_mask'))
                except Exception as e:
                    print(f"[Warn] Jetson (Val) Error: {e}");
                    edge_fail_info["Jetson"] = (edge_fail_info["Jetson"] + f" [val]{e}").strip()
                    Results_Jetson_val.append((0.0, 999.0, 0.0, 0.0))

                Results_CPU_test.append(
                    evaluate_local_hardware(model, data_trained, device_type='cpu', runs=args.cpu_runs, mask_type='test_mask'))
                if torch.cuda.is_available():
                    Results_GPU_test.append(
                        evaluate_local_hardware(model, data_trained, device_type='gpu', runs=args.gpu_runs, mask_type='test_mask'))
                try:
                    Results_KV260_test.append(
                        evaluate_remote_hardware(model, data_trained, dname, args.index, KV260_IP, KV260_USER, "kv260",
                                                 runs=args.kv260_runs, mask_type='test_mask'))
                except Exception as e:
                    print(f"[Warn] KV260 (Test) Error: {e}");
                    edge_fail_info["KV260"] = (edge_fail_info["KV260"] + f" [test]{e}").strip()
                    Results_KV260_test.append((0.0, 999.0, 0.0, 0.0))
                try:
                    Results_Jetson_test.append(
                        evaluate_remote_hardware(model, data_trained, dname, args.index, JETSON_IP, JETSON_USER,
                                                 "jetson", runs=args.jetson_runs, mask_type='test_mask'))
                except Exception as e:
                    print(f"[Warn] Jetson (Test) Error: {e}");
                    edge_fail_info["Jetson"] = (edge_fail_info["Jetson"] + f" [test]{e}").strip()
                    Results_Jetson_test.append((0.0, 999.0, 0.0, 0.0))

        # =========================================================================
        # 统计平均值 (区分 Val 和 Test)
        # =========================================================================
        test_acc_mean, val_acc_mean = np.mean(Results0, axis=0) * 100
        test_acc_std = np.sqrt(np.var(Results0, axis=0)[0]) * 100

        # Val 硬件均值
        cpu_val_avg = np.mean(Results_CPU_val, axis=0) if Results_CPU_val else [0, 999, 0, 0]
        gpu_val_avg = np.mean(Results_GPU_val, axis=0) if Results_GPU_val else [0, 999, 0, 0]
        kv260_val_avg = np.mean(Results_KV260_val, axis=0) if Results_KV260_val else [0, 999, 0, 0]
        jetson_val_avg = np.mean(Results_Jetson_val, axis=0) if Results_Jetson_val else [0, 999, 0, 0]
        kv260_val_lat_std = np.std([r[1] for r in Results_KV260_val]) if len(Results_KV260_val) > 1 else 0.0

        # Test 硬件均值
        cpu_test_avg = np.mean(Results_CPU_test, axis=0) if Results_CPU_test else [0, 999, 0, 0]
        gpu_test_avg = np.mean(Results_GPU_test, axis=0) if Results_GPU_test else [0, 999, 0, 0]
        kv260_test_avg = np.mean(Results_KV260_test, axis=0) if Results_KV260_test else [0, 999, 0, 0]
        jetson_test_avg = np.mean(Results_Jetson_test, axis=0) if Results_Jetson_test else [0, 999, 0, 0]
        kv260_test_lat_std = np.std([r[1] for r in Results_KV260_test]) if len(Results_KV260_test) > 1 else 0.0

        avg_train_time = total_train_time / args.RPMAX

        # 打印清晰的终端日志
        print(f"++++++++++++++++++++++++{args.index}++++++++++++++++++++++++++++++++++++++")
        print(f'LLM_HSGNAS on dataset {args.dataset}, in {args.RPMAX} repeated experiment:')
        print(f'GPU 验证集平均准确率 = {val_acc_mean:.4f}% \t 测试集平均准确率 = {test_acc_mean:.4f}%')
        print(f'--- [评估集 VAL] 硬件表现 ---')
        print(
            f'[CPU]    Acc={cpu_val_avg[0] * 100:.2f}% \t Lat={cpu_val_avg[1]:.6f}s \t Thr={cpu_val_avg[2]:.2f} \t Ene={cpu_val_avg[3]:.6f}J')
        print(
            f'[GPU]    Acc={gpu_val_avg[0] * 100:.2f}% \t Lat={gpu_val_avg[1]:.6f}s \t Thr={gpu_val_avg[2]:.2f} \t Ene={gpu_val_avg[3]:.6f}J')
        print(
            f'[KV260]  Acc={kv260_val_avg[0] * 100:.2f}% \t Lat={kv260_val_avg[1]:.6f}s \t Thr={kv260_val_avg[2]:.2f} \t Ene={kv260_val_avg[3]:.6f}J')
        print(
            f'[Jetson] Acc={jetson_val_avg[0] * 100:.2f}% \t Lat={jetson_val_avg[1]:.6f}s \t Thr={jetson_val_avg[2]:.2f} \t Ene={jetson_val_avg[3]:.6f}J')
        print(f'--- [测试集 TEST] 硬件表现 ---')
        print(
            f'[CPU]    Acc={cpu_test_avg[0] * 100:.2f}% \t Lat={cpu_test_avg[1]:.6f}s \t Thr={cpu_test_avg[2]:.2f} \t Ene={cpu_test_avg[3]:.6f}J')
        print(
            f'[GPU]    Acc={gpu_test_avg[0] * 100:.2f}% \t Lat={gpu_test_avg[1]:.6f}s \t Thr={gpu_test_avg[2]:.2f} \t Ene={gpu_test_avg[3]:.6f}J')
        print(
            f'[KV260]  Acc={kv260_test_avg[0] * 100:.2f}% \t Lat={kv260_test_avg[1]:.6f}s \t Thr={kv260_test_avg[2]:.2f} \t Ene={kv260_test_avg[3]:.6f}J')
        print(
            f'[Jetson] Acc={jetson_test_avg[0] * 100:.2f}% \t Lat={jetson_test_avg[1]:.6f}s \t Thr={jetson_test_avg[2]:.2f} \t Ene={jetson_test_avg[3]:.6f}J')
        print(f"==========================================================================")

        # -------------------------------------------------------------------------
        # 📁 文件 1：写入 generate_pop.json (只给大模型看 Validation 指标！)
        # -------------------------------------------------------------------------
        with open(JSON_file, 'r+') as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            content = f.read().strip()
            data_list = json.loads(content) if content else []

            if len(data_list) > args.index:

                ##
                # 替换原来的 objective = np.round(val_acc_mean, 5)
                # ===== 多目标联合 reward（替代原 val_acc） =====
                kv260 = kv260_val_avg  # [acc, lat, thr, energy]

                # 各数据集参数配置
                config = {
                    "Cora": {"REF": (0.8201, 0.5077, 11616, 2815.9), "W": (0.88, 0.12)},
                    "Cornell": {"REF": (0.6460, 0.0328, 19766, 182.7), "W": (0.87, 0.13)},
                    "Photo": {"REF": (0.9108, 8.5221, 20782, 46402.1), "W": (0.99, 0.01)},
                    "PubMed": {"REF": (0.8318, 4.3479, 11973, 25941.5), "W": (0.98, 0.02)},
                    "Texas": {"REF": (0.8307, 0.0239, 12722, 132.3), "W": (0.70, 0.30)},
                }
                cfg = config[args.dataset]
                REF_ACC, REF_LAT, REF_THR, REF_ENE = cfg["REF"]
                W_ACC, W_HW = cfg["W"]

                # 归一化
                norm_acc = min(kv260[0] / REF_ACC, 1.0)
                norm_lat = max(1.0 - kv260[1] / REF_LAT, 0.0)
                norm_thr = min(kv260[2] / REF_THR, 1.0)
                norm_ene = max(1.0 - kv260[3] / REF_ENE, 0.0)

                # 硬件合并
                norm_hw = (norm_lat + norm_thr + norm_ene) / 3.0
                combined = W_ACC * norm_acc + W_HW * norm_hw

                data_list[args.index]["objective"] = np.round(combined, 5)

                #3
                #data_list[args.index]["objective"] = np.round(val_acc_mean, 5)
                # ✅ CPU/GPU 正常但边缘设备部署失败时保留报错；无失败才清空
                err_parts = [v for v in edge_fail_info.values() if v]
                if err_parts:
                    data_list[args.index]["error_msg"] = "Runtime/Deployment Error: " + "; ".join(err_parts)[:600]
                else:
                    data_list[args.index]["error_msg"] = ""

                # 仅将 Val 的结果赋给 hw_metrics
                data_list[args.index]["hw_metrics"] = {
                    "acc": np.round(kv260_val_avg[0], 5),
                    "lat": np.round(kv260_val_avg[1], 6),
                    "lat_std": np.round(kv260_val_lat_std, 6),
                    "thr": np.round(kv260_val_avg[2], 2),
                    "energy": np.round(kv260_val_avg[3], 5),
                    "devices": {
                        "CPU": {"acc": cpu_val_avg[0], "lat": cpu_val_avg[1], "thr": cpu_val_avg[2],
                                "energy": cpu_val_avg[3]},
                        "GPU": {"acc": gpu_val_avg[0], "lat": gpu_val_avg[1], "thr": gpu_val_avg[2],
                                "energy": gpu_val_avg[3]},
                        "KV260": {"acc": kv260_val_avg[0], "lat": kv260_val_avg[1], "thr": kv260_val_avg[2],
                                  "energy": kv260_val_avg[3]},
                        "Jetson": {"acc": jetson_val_avg[0], "lat": jetson_val_avg[1], "thr": jetson_val_avg[2],
                                   "energy": jetson_val_avg[3]}
                    }
                }
                f.seek(0)
                f.truncate()
                json.dump(data_list, f, indent=4, ensure_ascii=False)
            fcntl.flock(f, fcntl.LOCK_UN)

        # -------------------------------------------------------------------------
        # 📁 文件 2：记录详尽 Log 到最终结果文件 (保存所有双端数据和代码！)
        # -------------------------------------------------------------------------
        LOG_PATH = os.path.join(ROOT_PATH, "result", f"{args.dataset}_test_results_hw_final.json")
        record = {
            "index": args.index, "dataset": args.dataset,
            "val_acc_mean": np.round(val_acc_mean, 5),
            "test_acc_mean": np.round(test_acc_mean, 5),
            "test_acc_std": np.round(test_acc_std, 5),
            # 存下两套硬件跑分供写论文参考
            "hw_metrics_val": data_list[args.index]["hw_metrics"] if len(data_list) > args.index else {},
            "hw_metrics_test": {
                "acc": np.round(kv260_test_avg[0], 5),
                "lat": np.round(kv260_test_avg[1], 6),
                "lat_std": np.round(kv260_test_lat_std, 6),
                "thr": np.round(kv260_test_avg[2], 2),
                "energy": np.round(kv260_test_avg[3], 5),
                "devices": {
                    "CPU": {"acc": cpu_test_avg[0], "lat": cpu_test_avg[1], "thr": cpu_test_avg[2],
                            "energy": cpu_test_avg[3]},
                    "GPU": {"acc": gpu_test_avg[0], "lat": gpu_test_avg[1], "thr": gpu_test_avg[2],
                            "energy": gpu_test_avg[3]},
                    "KV260": {"acc": kv260_test_avg[0], "lat": kv260_test_avg[1], "thr": kv260_test_avg[2],
                              "energy": kv260_test_avg[3]},
                    "Jetson": {"acc": jetson_test_avg[0], "lat": jetson_test_avg[1], "thr": jetson_test_avg[2],
                               "energy": jetson_test_avg[3]}
                }
            },
            "avg_train_time_seconds": np.round(avg_train_time, 4),
            "code": data_list[args.index]["code"] if len(data_list) > args.index else ""
        }

        if not os.path.exists(LOG_PATH):
            with open(LOG_PATH, 'w') as f:
                json.dump([], f)

        with open(LOG_PATH, 'r+') as ext_file:
            fcntl.flock(ext_file, fcntl.LOCK_EX)
            content = ext_file.read().strip()
            try:
                extra_data = json.loads(content) if content else []
            except json.JSONDecodeError:
                extra_data = []

            extra_data.append(record)
            ext_file.seek(0)
            ext_file.truncate()
            json.dump(extra_data, ext_file, indent=4, ensure_ascii=False)
            fcntl.flock(ext_file, fcntl.LOCK_UN)

        print(f"✅ 架构 {args.index} (双规模式)评估完毕。Val延迟: {kv260_val_avg[1]:.6f}s")

    # =========================================================================
    # ✅ 终极异常拦截器 (将异常保存进 error_msg 并永久归档)
    # =========================================================================
    except Exception as e:
        error_msg = f"Runtime/Deployment Error: {str(e)}"
        print(f"\n⚠️ 拦截报错: 架构 {args.index} 运行失败 ({error_msg}). 已自动赋予 0 分并跳过。")
        code_str = ""
        try:
            with open(JSON_file, 'r+') as f:
                fcntl.flock(f, fcntl.LOCK_EX)
                content = f.read().strip()
                data_list = json.loads(content) if content else []
                if len(data_list) > args.index:
                    code_str = data_list[args.index].get("code", "")
                    data_list[args.index]["objective"] = 0.0
                    data_list[args.index]["error_msg"] = error_msg
                    data_list[args.index]["hw_metrics"] = {
                        "acc": 0.0, "lat": 999.0, "lat_std": 0.0, "thr": 0.0, "energy": 999.0,
                        "devices": {
                            "CPU": {"acc": 0.0, "lat": 999.0, "thr": 0.0, "energy": 999.0},
                            "GPU": {"acc": 0.0, "lat": 999.0, "thr": 0.0, "energy": 999.0},
                            "KV260": {"acc": 0.0, "lat": 999.0, "thr": 0.0, "energy": 999.0},
                            "Jetson": {"acc": 0.0, "lat": 999.0, "thr": 0.0, "energy": 999.0}
                        }
                    }
                    f.seek(0)
                    f.truncate()
                    json.dump(data_list, f, indent=4, ensure_ascii=False)
                fcntl.flock(f, fcntl.LOCK_UN)
        except Exception:
            pass

        # ✅ 新增：把硬件部署崩溃永久存档
        LOG_PATH = os.path.join(ROOT_PATH, "result", f"{args.dataset}_test_results_hw_final.json")
        record = {
            "index": args.index, "dataset": args.dataset,
            "val_acc_mean": 0.0, "test_acc_mean": 0.0, "test_acc_std": 0.0,
            "error_msg": error_msg,
            "hw_metrics_val": {}, "hw_metrics_test": {}, "code": code_str
        }
        if not os.path.exists(LOG_PATH):
            with open(LOG_PATH, 'w') as f: json.dump([], f)
        with open(LOG_PATH, 'r+') as ext_file:
            fcntl.flock(ext_file, fcntl.LOCK_EX)
            try:
                extra_data = json.loads(ext_file.read().strip())
            except json.JSONDecodeError:
                extra_data = []
            extra_data.append(record)
            ext_file.seek(0)
            ext_file.truncate()
            json.dump(extra_data, ext_file, indent=4, ensure_ascii=False)
            fcntl.flock(ext_file, fcntl.LOCK_UN)

        sys.exit(0)