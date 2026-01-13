import sys
import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TORCH_CUDA_ARCH_LIST"] = "12.0"
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline, set_seed
from transformers import AutoModelForCausalLM, AutoTokenizer
import socket
import json
from vllm import LLM, SamplingParams
from typing import Union, List
from functools import lru_cache

def check_internet(host="8.8.8.8", port=53, timeout=3):
    try:
        socket.setdefaulttimeout(timeout)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
        return True
    except Exception:
        return False

@lru_cache(maxsize=10000)
def format_prompt(tokenizer, msg:  str) -> str:
    message = [{"role": "user", "content": msg}]
    return tokenizer.apply_chat_template(
        message, 
        tokenize=False, 
        add_generation_prompt=True,
        enable_thinking=False
    )
# 本地模型路径
LOCAL_MODEL_PATH = "./local-model/hurrylu/Qwen3-0.6B-ADDR64V1"
DATASET_PATH = "/app/train.json"

# --- 网络连通性测试 ---
internet_ok = check_internet()
print("【Internet Connectivity Test】:",
      "CONNECTED" if internet_ok else "OFFLINE / BLOCKED")

# --- 模型加载（从本地加载，无需网络）---
print(f"从本地加载模型：{LOCAL_MODEL_PATH}")
################################### 初始化部分 ###################################
# 1. 设置路径
model_dir = LOCAL_MODEL_PATH

tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

# 2. 准备 Prompt 预热推理
json_path = DATASET_PATH
with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)
input_text = [item["instruction"] for item in data]
prompt_list = [format_prompt(tokenizer, item) for item in input_text]
warmup_prompt_list = prompt_list * 3


# 3. 配置采样参数 (SamplingParams)
sampling_params = SamplingParams(
    temperature=0,
    top_k=1,
    # top_p=0.8,
    max_tokens=256,
    stop=["\n\n", "<|endoftext|>", "<|im_end|>"],
    stop_token_ids=[tokenizer.eos_token_id],
)

# 4. 初始化 vLLM 引擎
llm = LLM(
    model=model_dir,
    dtype="bfloat16",
    quantization="fp8",         
    # kv_cache_dtype="fp8",
    trust_remote_code=True,
    tensor_parallel_size=1,
    gpu_memory_utilization=0.95,
    enforce_eager=False,
    max_model_len=1024,
    max_num_seqs=1024,
    enable_prefix_caching=True,
    disable_log_stats=True,    
)

# 5. 执行预热推理

print(f"开始预热推理")
outputs = llm.generate(warmup_prompt_list, sampling_params)

set_seed(42)

# --- API 定义 ---
# 创建FastAPI应用实例
app = FastAPI(
    title="Simple Inference Server",
    description="A simple API to run a small language model."
)

# 定义API请求的数据模型
class PromptRequest(BaseModel):
    prompt: Union[str, List[str]]

# 定义API响应的数据模型
class PredictResponse(BaseModel):
    response: Union[str, List[str]]
    
# --- API 端点 ---
@app.post("/predict", response_model=PredictResponse)
async def predict(request: PromptRequest):
    """
    接收一个prompt，使用加载的模型进行推理，并返回结果。
    """

    ################################### 正式推理部分 ###################################
    if isinstance(request.prompt, str):
        real_input_list = [request.prompt]
    else:
        real_input_list = request.prompt
    
    final_prompt_texts = [format_prompt(tokenizer,msg) for msg in real_input_list]
    # vLLM 自动进行高效的批量推理
    outputs = llm.generate(final_prompt_texts, sampling_params)
    
    generated = [output.outputs[0].text for output in outputs]
    print(f"成功生成 {len(generated)} 条结果")
    # print(generated)


    return PredictResponse(response=generated)


@app.get("/")
def health_check():
    """
    健康检查端点，用于确认服务是否启动成功。
    """
    return {"status": "batch"}