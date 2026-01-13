#!/bin/bash

# ================= 配置区域 =================
IMAGE_NAME="my-submit-image:v1"
CONTAINER_NAME="vllm_submission_test"
PORT=8000
DATA_FILE="train_base.json"
EXPANDED_FILE="train_expanded_300.json"
# ===========================================

# 颜色定义
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}[1/6] 环境清理...${NC}"
# 停止并删除旧容器，杀掉占用端口的进程
docker rm -f $CONTAINER_NAME 2>/dev/null
sudo lsof -i :$PORT | awk 'NR>1 {print $2}' | xargs -r sudo kill -9
echo "端口 $PORT 已清理。"

echo -e "${YELLOW}[2/6] 构建 Docker 镜像...${NC}"
# 使用 host 网络避免下载模型超时
docker build --network host -t $IMAGE_NAME .
if [ $? -ne 0 ]; then
    echo -e "${RED}镜像构建失败，脚本终止。${NC}"
    exit 1
fi

echo -e "${YELLOW}[3/6] 启动容器...${NC}"
docker run -d --gpus all --name $CONTAINER_NAME -p $PORT:$PORT $IMAGE_NAME

echo -e "${YELLOW}[4/6] 等待服务上线 (Health Check)...${NC}"
# 循环检查端口，最多等待 300 秒 (5分钟)
MAX_RETRIES=60
COUNT=0
URL="http://localhost:$PORT"

while [ $COUNT -lt $MAX_RETRIES ]; do
    # 尝试访问根路径或 health 端点
    STATUS=$(curl -s -o /dev/null -w "%{http_code}" $URL/)
    if [ "$STATUS" == "200" ] || [ "$STATUS" == "404" ]; then
        echo -e "${GREEN}服务已上线! HTTP 状态码: $STATUS${NC}"
        break
    fi
    echo "正在等待服务启动... ($COUNT/$MAX_RETRIES)"
    sleep 5
    let COUNT=COUNT+1
done

if [ $COUNT -eq $MAX_RETRIES ]; then
    echo -e "${RED}服务启动超时，请检查 docker logs $CONTAINER_NAME${NC}"
    docker logs --tail 20 $CONTAINER_NAME
    exit 1
fi

# GET 测试
echo -e "执行 GET 测试..."
curl -v $URL/
echo -e "\n---------------------------------"

echo -e "${YELLOW}[5/6] 准备测试数据 (100 -> 300)...${NC}"
# 使用嵌入式 Python 脚本处理 JSON 扩充
python3 -c "
import json
import sys

try:
    with open('$DATA_FILE', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 复制数据直到达到 300 条
    expanded_data = (data * 4)[:300]
    
    with open('$EXPANDED_FILE', 'w', encoding='utf-8') as f:
        json.dump(expanded_data, f, ensure_ascii=False, indent=2)
    
    print(f'成功生成 {len(expanded_data)} 条测试数据到 $EXPANDED_FILE')
except FileNotFoundError:
    print('错误: 找不到 $DATA_FILE')
    sys.exit(1)
"
if [ $? -ne 0 ]; then exit 1; fi

echo -e "${YELLOW}[6/6] 开始 Batch 并发测试与结果分析...${NC}"
# 嵌入式 Python 评测脚本
cat <<EOF > run_test.py
import json
import requests
import time
import difflib

URL = "http://localhost:$PORT/predict"
DATA_FILE = "$EXPANDED_FILE"

def calculate_similarity(a, b):
    """计算两个字符串的相似度 (0.0 - 1.0)"""
    return difflib.SequenceMatcher(None, str(a), str(b)).ratio()

def main():
    # 1. 加载数据
    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    prompts = [item.get("instruction", "") for item in data]
    references = [item.get("output", "") for item in data]
    
    print(f"🚀 发送 Batch 请求 (数量: {len(prompts)})...")
    
    # 2. 发送请求
    start_time = time.time()
    try:
        # 注意：Payload key 使用 'prompt' 对应列表，符合你之前的成功测试
        response = requests.post(URL, json={"prompt": prompts}, timeout=300)
        end_time = time.time()
        
        if response.status_code != 200:
            print(f"❌ 请求失败: {response.status_code}")
            print(response.text)
            return

        res_data = response.json()
        answers = res_data.get("response", [])
        
        total_time = end_time - start_time
        
        # 3. 统计分析
        if len(answers) != len(prompts):
            print(f"⚠️ 警告: 返回数量不一致! 发送 {len(prompts)}, 接收 {len(answers)}")
            return

        total_chars = sum(len(str(a)) for a in answers)
        speed = total_chars / total_time
        
        print("\n" + "="*40)
        print(f"📊 性能报告")
        print(f"总耗时: {total_time:.4f} 秒")
        print(f"总字符: {total_chars}")
        print(f"估算速度: {speed:.2f} chars/s (注意: 评测系统可能按此计算)")
        print("="*40)
        
        # 4. 准确率/区别度分析
        print("\n🔍 准确度抽样分析 (前 5 条 + 关键公式检测)")
        low_sim_count = 0
        blockidx_check = False
        
        for i in range(len(answers)):
            gen = str(answers[i]).strip()
            ref = str(references[i]).strip()
            sim = calculate_similarity(gen, ref)
            
            # 检查关键 CUDA 公式 (blockIdx)
            if "blockIdx" in ref and "blockIdx" in gen:
                blockidx_check = True
            
            if sim < 0.23: # 基础题及格线
                low_sim_count += 1
            
            # 只打印前 5 条或者相似度极低的数据
            if i < 5:
                print(f"【题目 {i+1}】相似度: {sim:.2f}")
                print(f"🔴 [生成]: {gen[:100]}...")
                print(f"🟢 [参考]: {ref[:100]}...")
                print("-" * 30)

        print("="*40)
        print(f"相似度 < 0.23 (不及格风险) 数量: {low_sim_count}/{len(answers)}")
        
        if blockidx_check:
             print("✅ CUDA 公式检查: 通过 (检测到 blockIdx)")
        else:
             # 如果参考答案里本来就没有公式题，这里会误报，但作为一个提醒是好的
             if any("blockIdx" in r for r in references):
                 print("❌ CUDA 公式检查: 失败! (参考答案有 blockIdx 但生成答案未检测到)")
             else:
                 print("ℹ️ 本次测试数据中未包含 CUDA 索引公式题")

    except Exception as e:
        print(f"❌ 发生异常: {e}")

if __name__ == "__main__":
    main()
EOF

# 运行评测
python3 run_test.py

# 清理临时文件
rm run_test.py
# rm $EXPANDED_FILE  # 如果你想保留生成的300条数据，注释掉这行