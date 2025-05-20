import platform
import psutil
import torch
import json
import os
from openai import OpenAI  # 引入 OpenAI 类

# OpenAI API 配置
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')  # 替换为实际的 OpenAI API 密钥
BASE_URL = "https://api.openai.com/v1"  # OpenAI API 的基础 URL

def get_hardware_profile():
    """极简硬件探测基线"""
    info = {
        'cpu_cores': psutil.cpu_count(logical=True),
        'total_memory': psutil.virtual_memory().total // (1024**3),  # GB
        'cuda_available': torch.cuda.is_available(),
        'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
    }
    
    if torch.cuda.is_available():
        info['gpus'] = []
        for i in range(torch.cuda.device_count()):
            info['gpus'].append({
                'name': torch.cuda.get_device_properties(i).name,
                'memory': torch.cuda.get_device_properties(i).total_memory // (1024**3)
            })
            
    return info

def query_llm_framework(hardware, task_desc):
    """调用 OpenAI 的大模型决策"""
    system_prompt = """
你是一个硬件感知的并行计算框架决策系统。
当前分析目标：为计算任务选择最佳计算框架，从这些框架中选择："Serial","OpenMP","CUDA",
"""
    
    user_prompt_content = f"""
硬件配置：
- CPU核心：{hardware['cpu_cores']} 核
- 内存大小：{hardware['total_memory']} GB
- CUDA支持：{"是" if hardware['cuda_available'] else "否"}
- GPU数量：{hardware['gpu_count']} 台
- GPU规格：{json.dumps(hardware.get('gpus', []), ensure_ascii=False)}

任务需求：{task_desc}

请严格返回以下JSON格式：
{{"framework": "推荐框架", "reason": "推荐理由"}}
"""
    
    user_prompt = {
        "role": "user",
        "content": user_prompt_content
    }

    client = OpenAI(
        api_key=os.getenv('DASHSCOPE_API_KEY'),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )


    messages = [
        {"role": "system", "content": system_prompt},
        user_prompt
    ]

    completion = client.chat.completions.create(
        model="llama-4-scout-17b-16e-instruct",  
        messages=messages,
        temperature=0.2,
        max_tokens=2500,
        stop=["\n```\n"]
    )

    # 提取返回的 JSON 格式字符串
    output_text = completion.choices[0].message.content.strip()
    try:
        return json.loads(output_text)
    except json.JSONDecodeError:
        return {"framework": "Fallback", "reason": "返回内容不符合 JSON 格式"}

if __name__ == "__main__":
    # 示例用法
    hardware = get_hardware_profile()
    task_desc = "一个大小为1GB的图进行BFS"
    print("硬件指纹：", json.dumps(hardware, indent=2))
    
    result = query_llm_framework(hardware, task_desc)
    print("推荐框架：", result['framework'])
    print("推荐依据：", result['reason'])