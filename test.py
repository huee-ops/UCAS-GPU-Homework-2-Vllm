import json
import requests
import time

# 配置信息
URL = "http://localhost:8000/predict"
TEST_COUNT = 100  # 测试题目数量

def main():
    # 1. 从 train.json 加载数据
    try:
        with open('train_base.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print("错误: 未找到 train.json 文件")
        return

    # 取前 N 道题目
    subset = data[:TEST_COUNT]
    
    # 提取 instruction (题目) 和 output (参考答案)
    prompts = [item.get("instruction", "") for item in subset]
    references = [item.get("output", "") for item in subset]
    
    print(f"🚀 开始 Batch 测试，正在发送 {len(prompts)} 道题目...")

    # 2. 发送单次 POST 请求
    # 格式保持不变：Payload key 为 "prompt" (列表)，返回 key 为 "response" (列表)
    start_time = time.time()
    try:
        response = requests.post(URL, json={"prompt": prompts}, timeout=300) 
        end_time = time.time()

        if response.status_code == 200:
            res_data = response.json()
            # 获取返回的答案列表
            answers = res_data.get("response", []) 
            
            total_time = end_time - start_time
            print("\n" + "="*30)
            print(f"✅ 测试成功!")
            print(f"总耗时: {total_time:.2f} 秒")

            # 3. 性能估算
            if len(answers) == len(prompts):
                total_chars = sum(len(str(a)) for a in answers)
                print(f"生成字符总数: {total_chars}")
                print(f"估算速度: {total_chars / total_time:.2f} chars/s")
                
                print("="*30)
                print("🔍 答案对比 (生成 vs 参考) - 前 5 条示例 + 关键公式检查")
                print("="*30)
                
                # 4. 打印对比结果 (只打印前 10 条，避免刷屏，你可以改大 range)
                for i in range(len(answers)):
                    # 如果想看所有结果，把下面这行 if 去掉
                    if i < 5 or "blockIdx" in references[i]: 
                        gen_ans = str(answers[i]).strip()
                        ref_ans = str(references[i]).strip()
                        
                        print(f"【题目 {i+1}】: {prompts[i]}")
                        print(f"🔴 [生成]: {gen_ans[:200]}..." if len(gen_ans) > 200 else f"🔴 [生成]: {gen_ans}")
                        print(f"🟢 [参考]: {ref_ans[:200]}..." if len(ref_ans) > 200 else f"🟢 [参考]: {ref_ans}")
                        
                        # 简单的自动检查逻辑 (针对之前的公式错误)
                        if "blockIdx" in references[i] and "blockDim.y" in gen_ans:
                            print("⚠️ 警告: 检测到可能的公式错误 (使用了 blockDim.y 而非 blockIdx.x)！")
                        
                        print("-" * 50)
            else:
                print(f"⚠️ 警告: 返回答案数量 ({len(answers)}) 与题目数量 ({len(prompts)}) 不一致！")
            
            print("="*30)
        else:
            print(f"❌ 请求失败，状态码: {response.status_code}")
            print(f"错误日志: {response.text}")
            
    except Exception as e:
        print(f"❌ 发生异常: {str(e)}")

if __name__ == "__main__":
    main()