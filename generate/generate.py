import json
import os
from openai import OpenAI
from config import CONFIG

def check_available_devices(hardware):
    available_devices = []
    if "cpus" in hardware:
        for cpu in hardware["cpus"]:
            if cpu.get("available") == "True":
                available_devices.append({
                    "type": "CPU",
                    "cores": cpu.get("cores", "N/A"),
                    "threads": cpu.get("threads", "N/A"),
                    "frequency": cpu.get("frequency", "N/A")
                })
    if hardware.get("gpu", {}).get("available") == "True":
        available_devices.append({
            "type": "GPU",
            "cores": hardware["gpu"].get("cuda_cores", "N/A"),
            "memory": hardware["gpu"].get("memory", {})
        })
    return available_devices

def generate_code(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    global_output = {
        "tasks": []
    }

    available_devices = check_available_devices(config["hardware"])
    available_devices_info = []
    for device in available_devices:
        if device["type"] == "CPU":
            device_info = f"CPU（核心数: {device['cores']}, 线程数: {device['threads']}, 频率: {device['frequency']})"
        else:
            device_info = f"GPU（CUDA核心数: {device['cores']}, 显存: {device['memory'].get('size', 'N/A')})"
        available_devices_info.append(device_info)
    available_devices_info = ", ".join(available_devices_info)

    available_frameworks = ["Serial"]

    for task in config["tasks"]:
        system_prompt = f"""你是一个C++专家，需要根据以下配置生成优化的并行计算代码：
        - 任务类型: {task['type']}
        - 硬件配置: {available_devices_info}
        - 可用框架: {', '.join(available_frameworks)}
        请根据硬件信息和任务需求，从可选框架中选择最合适的一个来生成代码（必须选择一个），生成的代码尽可能利用所有硬件资源同时降低内存开销。如果选用Serial则禁止使用任何并行框架或者方法
        """

        user_prompt_content = f"根据任务生成优化的并行计算代码：\n\n"
        for framework in available_frameworks:
            if framework == "CUDA":
                function_signature = task["function_signatures"]["CUDA"]
                context = task["contexts"]["CUDA"]
            else:
                function_signature = task["function_signatures"]["other"]
                context = task["contexts"]["other"]
            user_prompt_content += f"如果选择 {framework} 框架：\n包含的头文件和结构体定义：\n{context}\n\n函数签名：\n{function_signature}\n\n"

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
            model=CONFIG["model"],
            messages=[
                {
                    "role": "system",
                    "content": f"请严格按以下格式输出代码：\n1. 代码必须用\n```cpp\n开始 用\n```\n结束，中间不得包含除C++代码外其他内容\n2. 只输出所选框架对应的函数实现，不输出prompt给出的结构体定义\n3. 禁止输出任何解释性文字或注释\n4. 在代码前指定所选择的框架（例如：选择的框架：<框架名称>）\n5. 尽可能根据硬件条件减小资源消耗\n6.在给出最终结果之前检查代码逻辑性，优化内存占用并修改。检查输出格式是否符合要求"
                },
                *messages
            ],
            temperature=0.2,
            max_tokens=2500,
            stop=["\n```\n"]
        )

        response_content = completion.choices[0].message.content
        response_lines = response_content.strip().split("\n")

        framework = None
        code_lines = []
        inside_code_block = False

        for line in response_lines:
            if line.startswith("```cpp"):
                inside_code_block = True
                continue
            elif line.startswith("```"):
                inside_code_block = False
                continue
            if inside_code_block:
                code_lines.append(line)

        if not code_lines:
            print("警告：未检测到代码内容，将跳过此任务")
            continue

        code_content = "\n".join(code_lines).strip()

        print(f"选择的框架：{framework}")
        print("*" * 50)
        print(f"代码实现：\n{code_content}")
        print("*" * 50)

        task_output = {
            "metadata": {
                "task_type": task["type"],
                "hardware": config["hardware"],
                "code": code_content,
                "framework": framework
            }
        }
        global_output["tasks"].append(task_output)

    with open("output.json", "w") as f:
        json.dump(global_output, f, indent=2, ensure_ascii=False)

    print("所有任务的代码生成完成，结果已保存到 output.json")

if __name__ == "__main__":
    generate_code("input.json")