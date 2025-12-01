from typing import Optional, Dict, Any, List
import os
import sys
import numpy as np
import threading
import uuid
import datetime
import traceback

from fastmcp import FastMCP

# Import LightAD modules
sys.path.insert(0, os.path.dirname(__file__))
from models.classifiers import KNN, decision_tree, MLP
from models.match import semantics_match
from models.optimizer import Bayes_optimizer
from utils import load_data, evaluation

mcp = FastMCP("lightad", debug=True, log_level="DEBUG")

# ---- Background task infrastructure ----
# In-memory task registry
TASKS: Dict[str, Dict[str, Any]] = {}
TASKS_LOCK = threading.Lock()

# Concurrency control (configure via env LIGHTAD_MAX_CONCURRENT)
MAX_CONCURRENT = int(os.getenv("LIGHTAD_MAX_CONCURRENT", "2"))
TASKS_SEM = threading.Semaphore(MAX_CONCURRENT)

# China Standard Time (UTC+08:00)
TZ_CN = datetime.timezone(datetime.timedelta(hours=8))


def _now_iso() -> str:
    return datetime.datetime.now(TZ_CN).isoformat()


def _create_task(task_type: str, params: Dict[str, Any]) -> str:
    task_id = str(uuid.uuid4())
    task = {
        "id": task_id,
        "type": task_type,
        "params": params,
        "status": "queued",  # queued|running|succeeded|failed|cancelled
        "progress": 0.0,
        "created_at": _now_iso(),
        "started_at": None,
        "completed_at": None,
        "result": None,
        "error": None,
        "traceback": None,
    }
    with TASKS_LOCK:
        TASKS[task_id] = task
    return task_id


def _set_task(task_id: str, **updates):
    with TASKS_LOCK:
        if task_id in TASKS:
            TASKS[task_id].update(**updates)


def _get_task(task_id: str) -> Dict[str, Any]:
    with TASKS_LOCK:
        return dict(TASKS.get(task_id, {}))


def _list_tasks() -> List[Dict[str, Any]]:
    with TASKS_LOCK:
        return [dict(t) for t in TASKS.values()]


def _start_background(target, *args):
    t = threading.Thread(target=target, args=args, daemon=True)
    t.start()
    return t


# Worker functions
def _train_hdfs_worker(task_id: str, params: Dict[str, Any]):
    """Worker for HDFS dataset anomaly detection"""
    try:
        with TASKS_SEM:
            _set_task(task_id, status="running", started_at=_now_iso(), progress=0.1)
            
            model_name = params.get("model", "knn")
            eliminate = params.get("eliminate", False)
            
            # Load data
            _set_task(task_id, progress=0.2)
            loader = load_data(dataset="hdfs", eliminated=eliminate)
            x_train_arr, x_test_arr, y_train_arr, y_test_arr = loader.reload_data()
            
            # Process each split
            data_len = len(x_train_arr)
            results_all = []
            
            for i in range(data_len):
                progress = 0.3 + (i / data_len) * 0.6
                _set_task(task_id, progress=progress)
                
                x_train = x_train_arr[i].tolist()
                x_test = x_test_arr[i].tolist()
                y_train = y_train_arr[i].tolist()
                y_test = y_test_arr[i].tolist()
                
                # Train model
                if model_name == "knn":
                    params_dict = {"n_neighbors": params.get("n_neighbors", 1)}
                    labels_pre, train_time, infer_time = KNN(x_train, x_test, y_train, **params_dict)
                elif model_name == "dt":
                    params_dict = {}
                    labels_pre, train_time, infer_time = decision_tree(x_train, x_test, y_train, **params_dict)
                elif model_name == "slfn":
                    params_dict = {"hidden_layer_sizes": params.get("hidden_layer_sizes", (25,))}
                    labels_pre, train_time, infer_time = MLP(x_train, x_test, y_train, **params_dict)
                else:
                    raise ValueError(f"Unknown model: {model_name}")
                
                # Evaluate
                evaluate = evaluation(y_test, labels_pre)
                p, r, f, s, b = evaluate.calc_metrics()
                
                results_all.append({
                    "split": i,
                    "precision": float(p),
                    "recall": float(r),
                    "f1_score": float(f),
                    "specificity": float(s),
                    "balanced_acc": float(b),
                    "train_time": float(train_time),
                    "infer_time": float(infer_time),
                })
            
            # Calculate averages
            avg_results = {
                "precision": float(np.mean([r["precision"] for r in results_all])),
                "recall": float(np.mean([r["recall"] for r in results_all])),
                "f1_score": float(np.mean([r["f1_score"] for r in results_all])),
                "specificity": float(np.mean([r["specificity"] for r in results_all])),
                "balanced_acc": float(np.mean([r["balanced_acc"] for r in results_all])),
                "avg_train_time": float(np.mean([r["train_time"] for r in results_all])),
                "avg_infer_time": float(np.mean([r["infer_time"] for r in results_all])),
            }
            
            result = {
                "status": "ok",
                "model": model_name,
                "dataset": "hdfs",
                "eliminated": eliminate,
                "num_splits": data_len,
                "detailed_results": results_all,
                "average_results": avg_results,
            }
            
            _set_task(task_id, status="succeeded", progress=1.0, completed_at=_now_iso(), result=result)
            
    except Exception as ex:
        _set_task(task_id, status="failed", completed_at=_now_iso(), error=str(ex), traceback=traceback.format_exc())


def _train_super_worker(task_id: str, params: Dict[str, Any]):
    """Worker for supercomputer datasets"""
    try:
        with TASKS_SEM:
            _set_task(task_id, status="running", started_at=_now_iso(), progress=0.1)
            
            dataset = params.get("dataset", "bgl")
            dist_type = params.get("dist_type", "Jaccard")
            sample_ratio = params.get("sample_ratio", 0.1)
            window = params.get("window", 10)
            step = params.get("step", 10)
            
            # Load data
            _set_task(task_id, progress=0.2)
            loader = load_data(dataset=dataset)
            x_train, x_test, y_train, y_test = loader.reload_data()
            x_train, x_test, y_train, y_test = x_train[0], x_test[0], y_train[0], y_test[0]
            
            # Sample training data
            _set_task(task_id, progress=0.3)
            idx_list = np.random.choice(len(x_train), int(sample_ratio * len(x_train)), replace=False)
            xx_train = [x_train[idx] for idx in idx_list]
            yy_train = [y_train[idx] for idx in idx_list]
            
            # Run semantic matching
            _set_task(task_id, progress=0.4)
            model = semantics_match()
            labels_pre, matched, train_time, infer_time = model.run(xx_train, x_test, yy_train, dist_type=dist_type)
            
            # Evaluate with windows
            _set_task(task_id, progress=0.9)
            evaluate = evaluation(y_test, labels_pre)
            p, r, f, s, b = evaluate.calc_with_windows(window=window, step=step)
            
            result = {
                "status": "ok",
                "model": "semantics_match",
                "dataset": dataset,
                "dist_type": dist_type,
                "sample_ratio": sample_ratio,
                "window": window,
                "step": step,
                "train_samples": len(xx_train),
                "test_samples": len(x_test),
                "precision": float(p),
                "recall": float(r),
                "f1_score": float(f),
                "specificity": float(s),
                "balanced_acc": float(b),
                "train_time": float(train_time),
                "infer_time": float(infer_time),
            }
            
            _set_task(task_id, status="succeeded", progress=1.0, completed_at=_now_iso(), result=result)
            
    except Exception as ex:
        _set_task(task_id, status="failed", completed_at=_now_iso(), error=str(ex), traceback=traceback.format_exc())


def _optimize_model_worker(task_id: str, params: Dict[str, Any]):
    """Worker for model optimization"""
    try:
        with TASKS_SEM:
            _set_task(task_id, status="running", started_at=_now_iso(), progress=0.1)
            
            model_name = params.get("model", "knn")
            l1 = params.get("l1", 0.6)
            l2 = params.get("l2", 0.2)
            l3 = params.get("l3", 0.2)
            
            # Validate weights
            if abs(l1 + l2 + l3 - 1.0) > 0.01:
                raise ValueError("l1, l2, l3 must sum to 1.0")
            
            # Load deduplicated HDFS data
            _set_task(task_id, progress=0.2)
            loader = load_data(dataset="hdfs", eliminated=True)
            x_train_arr, x_test_arr, y_train_arr, y_test_arr = loader.reload_data()
            
            # Optimize on each split
            data_len = min(len(x_train_arr), 5)
            f_all = []
            t_time = []
            i_time = []
            best_params_list = []
            
            for i in range(data_len):
                progress = 0.3 + (i / data_len) * 0.6
                _set_task(task_id, progress=progress)
                
                x_train = x_train_arr[i].tolist()
                x_test = x_test_arr[i].tolist()
                y_train = y_train_arr[i].tolist()
                y_test = y_test_arr[i].tolist()
                
                # Define parameter ranges
                if model_name == "knn":
                    params_range_dict = {"n_neighbors": (1, 11), "metric": (0, 3)}
                    optimizer = Bayes_optimizer(l1, l2, l3, KNN, x_train, y_train, params_range_dict)
                    best_params = optimizer.optimize()
                    labels_pre, train_time, infer_time = KNN(x_train, x_test, y_train, **best_params)
                elif model_name == "dt":
                    params_range_dict = {
                        "criterion": (0, 2),
                        "min_samples_leaf": (1, 5),
                        "max_depth": (5, 70),
                        "min_samples_split": (2, 5),
                    }
                    optimizer = Bayes_optimizer(l1, l2, l3, decision_tree, x_train, y_train, params_range_dict)
                    best_params = optimizer.optimize()
                    labels_pre, train_time, infer_time = decision_tree(x_train, x_test, y_train, **best_params)
                elif model_name == "slfn":
                    params_range_dict = {
                        "hidden_neurons": (5, 100),
                        "solver": (0, 3),
                        "activation": (0, 4),
                        "alpha": (1e-6, 1e-1),
                        "tol": (1e-6, 1e-1),
                        "max_iter": (50, 400),
                    }
                    optimizer = Bayes_optimizer(l1, l2, l3, MLP, x_train, y_train, params_range_dict)
                    best_params = optimizer.optimize()
                    labels_pre, train_time, infer_time = MLP(x_train, x_test, y_train, **best_params)
                else:
                    raise ValueError(f"Unknown model: {model_name}")
                
                # Evaluate
                evaluate = evaluation(y_test, labels_pre)
                _, _, f, _, _ = evaluate.calc_metrics()
                
                f_all.append(f)
                t_time.append(train_time / len(x_train))
                i_time.append(infer_time / len(x_test))
                best_params_list.append(best_params)
            
            # Calculate ModelScore
            model_score = l1 * (np.mean(f_all) - 0.8) / 0.2 - l2 * np.mean(t_time) / 3e-2 - l3 * np.mean(i_time) / 2e-3
            
            result = {
                "status": "ok",
                "model": model_name,
                "weights": {"l1": l1, "l2": l2, "l3": l3},
                "model_score": float(model_score),
                "avg_f1_score": float(np.mean(f_all)),
                "avg_train_time_per_sample": float(np.mean(t_time)),
                "avg_infer_time_per_sample": float(np.mean(i_time)),
                "best_params_per_split": best_params_list,
            }
            
            _set_task(task_id, status="succeeded", progress=1.0, completed_at=_now_iso(), result=result)
            
    except Exception as ex:
        _set_task(task_id, status="failed", completed_at=_now_iso(), error=str(ex), traceback=traceback.format_exc())


def _preprocess_worker(task_id: str, params: Dict[str, Any]):
    """Worker for data preprocessing"""
    try:
        with TASKS_SEM:
            _set_task(task_id, status="running", started_at=_now_iso(), progress=0.1)
            
            dataset = params.get("dataset", "hdfs")
            eliminate = params.get("eliminate", False)
            train_ratio = params.get("train_ratio", 0.8)
            
            _set_task(task_id, progress=0.2)
            
            if dataset == "hdfs":
                loader = load_data(dataset="hdfs", eliminated=eliminate)
                times = 1 if eliminate else 5
                loader.load_and_split_hdfs(train_ratio=train_ratio, times=times)
            elif dataset in ["spirit", "tbird", "liberty", "bgl"]:
                loader = load_data(dataset=dataset)
                data_ranges = {
                    "spirit": [0, 7983345],
                    "tbird": [0, 10000000],
                    "liberty": [0, 10000000],
                    "bgl": [0, 4747963],
                }
                loader.load_and_split(data_range=data_ranges[dataset], train_ratio=train_ratio)
            else:
                raise ValueError(f"Unknown dataset: {dataset}")
            
            result = {
                "status": "ok",
                "dataset": dataset,
                "eliminated": eliminate if dataset == "hdfs" else None,
                "train_ratio": train_ratio,
                "message": f"Preprocessing completed for {dataset}",
            }
            
            _set_task(task_id, status="succeeded", progress=1.0, completed_at=_now_iso(), result=result)
            
    except Exception as ex:
        _set_task(task_id, status="failed", completed_at=_now_iso(), error=str(ex), traceback=traceback.format_exc())


# MCP Tools
@mcp.tool()
def lightad_train_hdfs(
    model: str = "knn",
    eliminate: bool = False,
    n_neighbors: int = 1,
    hidden_layer_sizes: tuple = (25,),
) -> Dict[str, Any]:
    """
    在 HDFS 数据集上训练异常检测模型（异步任务）。
    
    参数:
    - model: str = "knn"
        模型类型，可选："knn" (K-最近邻)、"dt" (决策树)、"slfn" (单层前馈神经网络)
    - eliminate: bool = False
        是否使用去重数据集
    - n_neighbors: int = 1
        KNN 模型的邻居数（仅当 model="knn" 时生效）
    - hidden_layer_sizes: tuple = (25,)
        SLFN 模型的隐藏层大小（仅当 model="slfn" 时生效）
    
    返回:
    - dict: {
        "status": "queued",
        "task_id": str,
        "type": "train_hdfs",
        "message": str,
        "estimated_time": str,
        "polling_advice": str
      }
    
    使用方法:
    1. 调用此函数获取 task_id
    2. 等待至少 30 秒（任务通常需要 2-10 分钟）
    3. 使用 get_task(task_id) 查询状态和结果
    4. 如果状态为 "running"，请等待更长时间后再查询，避免频繁轮询
    
    注意：任务在后台持续运行，不会因等待而停止。频繁查询会消耗对话轮次。
    """
    if model not in ["knn", "dt", "slfn"]:
        return {"status": "failed", "error": "model must be one of: knn, dt, slfn"}
    
    params = {
        "model": model,
        "eliminate": eliminate,
        "n_neighbors": n_neighbors,
        "hidden_layer_sizes": hidden_layer_sizes,
    }
    
    task_id = _create_task("train_hdfs", params)
    _start_background(_train_hdfs_worker, task_id, params)
    
    return {
        "status": "queued",
        "task_id": task_id,
        "type": "train_hdfs",
        "message": "任务已创建并在后台运行。此任务通常需要 2-10 分钟完成。",
        "estimated_time": "2-10 minutes",
        "polling_advice": "建议等待至少 30 秒后再查询进度，避免频繁轮询。任务在后台持续运行，即使不查询也不会停止。"
    }


@mcp.tool()
def lightad_train_super(
    dataset: str = "bgl",
    dist_type: str = "Jaccard",
    sample_ratio: float = 0.1,
    window: int = 10,
    step: int = 10,
) -> Dict[str, Any]:
    """
    在超级计算机数据集上进行语义匹配异常检测（异步任务）。
    
    参数:
    - dataset: str = "bgl"
        数据集名称，可选："bgl"、"spirit"、"tbird"、"liberty"
    - dist_type: str = "Jaccard"
        距离度量类型
    - sample_ratio: float = 0.1
        训练数据采样比例（0.0 ~ 1.0）
    - window: int = 10
        评估窗口大小
    - step: int = 10
        评估窗口步长
    
    返回:
    - dict: {
        "status": "queued",
        "task_id": str,
        "type": "train_super",
        "message": str,
        "estimated_time": str,
        "polling_advice": str
      }
    
    使用方法:
    1. 调用此函数获取 task_id
    2. 等待至少 20 秒（任务通常需要 1-5 分钟）
    3. 使用 get_task(task_id) 查询状态和结果
    4. 如果状态为 "running"，请等待更长时间后再查询，避免频繁轮询
    
    注意：任务在后台持续运行，不会因等待而停止。频繁查询会消耗对话轮次。
    """
    if dataset not in ["bgl", "spirit", "tbird", "liberty"]:
        return {"status": "failed", "error": "dataset must be one of: bgl, spirit, tbird, liberty"}
    
    if not 0.0 < sample_ratio <= 1.0:
        return {"status": "failed", "error": "sample_ratio must be between 0.0 and 1.0"}
    
    params = {
        "dataset": dataset,
        "dist_type": dist_type,
        "sample_ratio": sample_ratio,
        "window": window,
        "step": step,
    }
    
    task_id = _create_task("train_super", params)
    _start_background(_train_super_worker, task_id, params)
    
    return {
        "status": "queued",
        "task_id": task_id,
        "type": "train_super",
        "message": "任务已创建并在后台运行。此任务通常需要 1-5 分钟完成。",
        "estimated_time": "1-5 minutes",
        "polling_advice": "建议等待至少 20 秒后再查询进度，避免频繁轮询。任务在后台持续运行，即使不查询也不会停止。"
    }


@mcp.tool()
def lightad_optimize_model(
    model: str = "knn",
    l1: float = 0.6,
    l2: float = 0.2,
    l3: float = 0.2,
) -> Dict[str, Any]:
    """
    对模型进行贝叶斯优化，自动选择最优超参数（异步任务）。
    仅在去重的 HDFS 数据集上运行。
    
    参数:
    - model: str = "knn"
        模型类型，可选："knn"、"dt"、"slfn"
    - l1: float = 0.6
        模型准确率（F1-score）的相对重要性权重
    - l2: float = 0.2
        训练时间的相对重要性权重
    - l3: float = 0.2
        推理时间的相对重要性权重
    
    注意：l1 + l2 + l3 必须等于 1.0
    
    返回:
    - dict: {
        "status": "queued",
        "task_id": str,
        "type": "optimize",
        "message": str,
        "estimated_time": str,
        "polling_advice": str
      }
    
    使用方法:
    1. 调用此函数获取 task_id
    2. 等待至少 60 秒（贝叶斯优化通常需要 5-15 分钟）
    3. 使用 get_task(task_id) 查询优化结果
    4. 如果状态为 "running"，请等待更长时间后再查询，避免频繁轮询
    
    注意：任务在后台持续运行，不会因等待而停止。频繁查询会消耗对话轮次。
    """
    if model not in ["knn", "dt", "slfn"]:
        return {"status": "failed", "error": "model must be one of: knn, dt, slfn"}
    
    if abs(l1 + l2 + l3 - 1.0) > 0.01:
        return {"status": "failed", "error": "l1 + l2 + l3 must equal 1.0"}
    
    if any(x <= 0 for x in [l1, l2, l3]):
        return {"status": "failed", "error": "l1, l2, l3 must all be greater than 0"}
    
    params = {
        "model": model,
        "l1": l1,
        "l2": l2,
        "l3": l3,
    }
    
    task_id = _create_task("optimize", params)
    _start_background(_optimize_model_worker, task_id, params)
    
    return {
        "status": "queued",
        "task_id": task_id,
        "type": "optimize",
        "message": "任务已创建并在后台运行。贝叶斯优化通常需要 5-15 分钟完成。",
        "estimated_time": "5-15 minutes",
        "polling_advice": "建议等待至少 60 秒后再查询进度，避免频繁轮询。任务在后台持续运行，即使不查询也不会停止。"
    }


@mcp.tool()
def lightad_preprocess(
    dataset: str = "hdfs",
    eliminate: bool = False,
    train_ratio: float = 0.8,
) -> Dict[str, Any]:
    """
    预处理原始日志数据集（异步任务）。
    
    参数:
    - dataset: str = "hdfs"
        数据集名称，可选："hdfs"、"bgl"、"spirit"、"tbird"、"liberty"
    - eliminate: bool = False
        是否去重（仅对 HDFS 数据集有效）
    - train_ratio: float = 0.8
        训练集比例（0.0 ~ 1.0）
    
    返回:
    - dict: {
        "status": "queued",
        "task_id": str,
        "type": "preprocess",
        "message": str,
        "estimated_time": str,
        "polling_advice": str
      }
    
    使用方法:
    1. 确保原始数据在 datasets/original_datasets/ 目录下
    2. 调用此函数获取 task_id
    3. 等待至少 20 秒（数据预处理通常需要 1-3 分钟）
    4. 使用 get_task(task_id) 查询预处理状态
    5. 如果状态为 "running"，请等待更长时间后再查询，避免频繁轮询
    6. 预处理后的数据保存在 datasets/splited_datasets/ 目录下
    
    注意：任务在后台持续运行，不会因等待而停止。频繁查询会消耗对话轮次。
    """
    if dataset not in ["hdfs", "bgl", "spirit", "tbird", "liberty"]:
        return {"status": "failed", "error": "dataset must be one of: hdfs, bgl, spirit, tbird, liberty"}
    
    if not 0.0 < train_ratio < 1.0:
        return {"status": "failed", "error": "train_ratio must be between 0.0 and 1.0"}
    
    params = {
        "dataset": dataset,
        "eliminate": eliminate,
        "train_ratio": train_ratio,
    }
    
    task_id = _create_task("preprocess", params)
    _start_background(_preprocess_worker, task_id, params)
    
    return {
        "status": "queued",
        "task_id": task_id,
        "type": "preprocess",
        "message": "任务已创建并在后台运行。数据预处理通常需要 1-3 分钟完成。",
        "estimated_time": "1-3 minutes",
        "polling_advice": "建议等待至少 20 秒后再查询进度，避免频繁轮询。任务在后台持续运行，即使不查询也不会停止。"
    }


@mcp.tool()
def list_tasks() -> Dict[str, Any]:
    """
    列出所有后台任务。
    
    返回:
    - dict: {
        "count": int,
        "tasks": [任务对象列表]
      }
    """
    tasks = _list_tasks()
    return {"count": len(tasks), "tasks": tasks}


@mcp.tool()
def get_task(task_id: str) -> Dict[str, Any]:
    """
    查询指定任务的详细状态、进度和结果。
    
    参数:
    - task_id: str
        任务 ID（由创建任务的函数返回）
    
    返回:
    - dict: 完整任务对象，包含以下字段：
        - id: 任务ID
        - type: 任务类型
        - status: 状态（queued/running/succeeded/failed）
        - progress: 进度（0.0 ~ 1.0）
        - created_at: 创建时间
        - started_at: 开始时间
        - completed_at: 完成时间
        - result: 结果（成功时）
        - error: 错误信息（失败时）
        - traceback: 错误堆栈（失败时）
        - polling_advice: 查询建议（当任务正在运行时）
    
    重要提示：
    - 如果任务状态为 "running"，请等待至少 30-60 秒后再查询，避免频繁轮询
    - 任务在后台持续运行，不会因等待而停止
    - 频繁查询会消耗对话轮次，可能导致对话终止
    - 建议根据任务类型的预估时间合理设置查询间隔
    """
    task = _get_task(task_id)
    
    # 如果任务不存在
    if not task:
        return {
            "status": "not_found",
            "error": f"任务 {task_id} 不存在",
            "task_id": task_id
        }
    
    # 如果任务正在运行，添加查询建议
    if task.get("status") == "running":
        progress = task.get("progress", 0.0)
        task_type = task.get("type", "")
        
        # 根据任务类型和进度给出建议
        if task_type == "optimize":
            wait_time = "60-120 秒"
        elif task_type in ["train_hdfs", "train_super"]:
            wait_time = "30-60 秒"
        elif task_type == "preprocess":
            wait_time = "20-40 秒"
        else:
            wait_time = "30 秒"
        
        task["polling_advice"] = (
            f"任务正在运行中（进度: {progress:.1%}）。"
            f"建议等待至少 {wait_time} 后再查询，避免频繁轮询。"
            f"任务在后台持续运行，不会因等待而停止。"
        )
    
    return task


if __name__ == "__main__":
    # Start an SSE transport MCP server on port 2224 by default
    mcp.run(transport="sse", port=2224)

