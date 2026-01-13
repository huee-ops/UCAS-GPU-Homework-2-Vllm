# 2025年秋季国科大《GPU架构与编程》大作业二

**项目名称**：基于 vLLM 的 Qwen3-0.6B 大模型高性能推理服务  
**开源链接**：[入围决赛同学请在此处填写 Gitee/Github 链接，否则可删除]

---

## 1. 项目简介

本项目是一个针对并行科技赛道优化的大模型推理服务，基于 `vLLM` 推理引擎构建，旨在提供高吞吐、低延迟的文本生成能力。项目核心采用 `Qwen3-0.6B` 模型，并通过 `FastAPI` 封装为标准的 HTTP 接口，能够满足评测系统对 `/predict` 端点的高并发调用需求。

主要特性：
- **高性能引擎**：使用 vLLM 进行推理加速。
- **显存优化**：启用 FP8 量化与 Prefix Caching。
- **快速启动**：内置模型下载脚本与预热策略。

## 2. 技术方案与实现细节

### 2.1 环境构建
本项目基于并行科技推荐的 `vllm-openai:v0.11.0` 镜像构建。
- **基础镜像**: `swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/vllm/vllm-openai:v0.11.0`
- **依赖管理**: 通过 `requirements.txt` 安装 `modelscope` 等必要库，并配置了清华镜像源加速安装。

### 2.2 模型选型
- **模型**: `hurrylu/Qwen3-0.6B-ADDR64V1` (Qwen/Qwen3-0.6B 系列)。
- **加载方式**: 构建阶段使用 `download_model.py` 通过 ModelScope 下载权重至本地 `./local-model`，确保运行时无需联网。

### 2.3 核心优化 (serve.py)
在 `serve.py` 中实现了以下关键优化：
1.  **量化推理**: 启用 `quantization="fp8"`，在保持精度的同时大幅降低显存占用。
2.  **显存利用**: 设置 `gpu_memory_utilization=0.95`，充分利用 RTX 5090 的大显存。
3.  **Prefix Caching**: 开启 `enable_prefix_caching=True`，复用公共前缀的 KV Cache，提高多轮对话或相似 Prompt 的处理效率。
4.  **预热策略**: 服务启动前使用 `train.json` 中的数据进行预热 (`warmup`)，提前编译 CUDA Graph，消除首字延迟。

## 3. 项目结构

- `Dockerfile`: 容器构建配置文件，包含模型下载指令。
- `serve.py`: 推理服务核心代码，包含 vLLM 引擎初始化与 FastAPI 接口定义。
- `download_model.py`: 模型权重下载脚本。
- `requirements.txt`: Python 依赖列表。
- `train.json`: 用于服务启动时的预热数据。

## 4. 接口说明与运行指南

### 启动服务
容器启动命令（由 Dockerfile 指定）：
```bash
uvicorn serve:app --host 0.0.0.0 --port 8000
