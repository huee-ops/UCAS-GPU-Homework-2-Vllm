# 使用并行科技赛道推荐的 vllm 镜像
FROM swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/vllm/vllm-openai:v0.11.0

# 清除原有的ENTRYPOINT
ENTRYPOINT []

# 设置工作目录
WORKDIR /app

# 下载模型（利用构建缓存）
COPY download_model.py .
RUN python3 download_model.py

# 安装依赖
COPY requirements.txt .
RUN pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
RUN pip install --no-cache-dir -r requirements.txt

# 暴露端口
EXPOSE 8000

COPY . .

# 启动命令
CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000"]