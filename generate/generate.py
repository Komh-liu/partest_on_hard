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

def extract_framework_from_code(code_content):
    # Try to infer the framework from the code content
    if "#pragma omp" in code_content:
        return "OpenMP"
    elif "MPI_" in code_content:
        return "MPI"
    elif "__global__" in code_content:
        return "CUDA"
    elif "tbb::" in code_content:  # Add TBB detection
        return "TBB"
    else:
        return "Serial"

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
            device_info = f"CPU (Cores: {device['cores']}, Threads: {device['threads']}, Frequency: {device['frequency']})"
        else:
            device_info = f"GPU (CUDA Cores: {device['cores']}, Memory: {device['memory'].get('size', 'N/A')})"
        available_devices_info.append(device_info)
    available_devices_info = ", ".join(available_devices_info)

    available_frameworks = [
        "Serial",
        #"OpenMP",
        "TBB",  # Add TBB to available frameworks
        # "MPI",
        # "CUDA",
    ]

    for task in config["tasks"]:
        system_prompt = f"""You are a C++ expert. Generate optimized parallel computing code based on the following configuration:
        - Task type: {task['type']}
        - Hardware configuration: {available_devices_info}
        - Available frameworks: {', '.join(available_frameworks)}
        Please select the most suitable framework from the available ones according to the hardware information and task requirements (you must choose one). The generated code should utilize all hardware resources as much as possible while reducing memory overhead. If you choose Serial, do not use any parallel frameworks or methods.

        **Optimization Goals**:
        - Minimize memory usage.
        - Maximize CPU/GPU utilization.
        - Ensure thread safety and avoid race conditions.
        - **Consider parallel-friendly data structures**: Ensure that the chosen data structures can be efficiently utilized in a parallel environment to maximize performance. For example, use concurrent data structures or partition data to minimize contention and maximize parallelism.

        **Code Review and Correction**:
        - Check the logic of the generated code for correctness.
        - Ensure that the code adheres to best practices for the selected framework.
        - Modify the code if necessary to improve performance and reduce resource consumption.
        
        **Framework-Specific Information**:
        - Intel TBB is a C++ library for parallel programming that provides high-level abstractions for parallel patterns. It's designed for task parallelism and supports nested parallelism well.
        """
        
        user_prompt_content = f"Generate optimized parallel computing code according to the task:\n\n"
        for framework in available_frameworks:
            if framework == "CUDA":
                function_signature = task["function_signatures"]["CUDA"]
                context = task["contexts"]["CUDA"]
            else:
                function_signature = task["function_signatures"]["other"]
                context = task["contexts"]["other"]
            
            # Add TBB include headers for the TBB framework option
            if framework == "TBB":
                tbb_headers = """
                #include <tbb/tbb.h>
                #include <tbb/parallel_for.h>
                #include <tbb/parallel_reduce.h>
                #include <tbb/blocked_range.h>
                #include <tbb/concurrent_vector.h>
                #include <tbb/concurrent_queue.h>
                """
                context = tbb_headers + context
            
            user_prompt_content += f"If you choose the {framework} framework:\nIncluded header files and structure definitions:\n{context}\n\nFunction signature:\n{function_signature}\n\n"

        user_prompt_content += """
        **Optimization Instructions**:
        - Use efficient data structures to minimize memory footprint.
        - Ensure that the code is thread-safe and avoids race conditions.
        - Optimize loop structures to reduce overhead.
        - Use appropriate parallel constructs to maximize hardware utilization.
        - **Consider parallel-friendly data structures**: When selecting data structures, ensure they can be efficiently used in a parallel environment. For example, use concurrent data structures or partition data to minimize contention and maximize parallelism.
        
        **TBB-Specific Tips**:
        - Use `tbb::parallel_for` for parallel loops
        - Use `tbb::parallel_reduce` for reduction operations
        - Use `tbb::blocked_range` to define chunks of work
        - Consider `tbb::concurrent_vector` or `tbb::concurrent_queue` for shared data structures
        - Use `tbb::global_control` to set the maximum number of threads
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
            model=CONFIG["model"],
            messages=[
                {
                    "role": "system",
                    "content": f"Please output the code strictly in the following format:\n1. The code must start with\n```cpp\nand end with\n```\nThere should be no content other than C++ code in between.\n2. Only output the function implementation corresponding to the selected framework, do not output the structure definitions given in the prompt.\n3. Do not output any explanatory text or comments.\n4. Specify the selected framework before the code (e.g., Selected framework: <Framework name>).\n5. Try to reduce resource consumption according to the hardware conditions as much as possible.\n6. Check the logic of the code before giving the final result, optimize the memory usage and modify it. Check if the output format meets the requirements."
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

        # Extract framework information from the response
        for line in response_lines:
            if "Selected framework:" in line:
                framework_text = line.split("Selected framework:")[1].strip()
                # Clean up the framework name
                if framework_text == "Intel TBB" or framework_text == "TBB":
                    framework = "TBB"
                elif framework_text == "OpenMP":
                    framework = "OpenMP"
                elif framework_text == "CUDA":
                    framework = "CUDA"
                elif framework_text == "MPI":
                    framework = "MPI"
                elif "Serial" in framework_text:
                    framework = "Serial"
            
            if line.startswith("```cpp"):
                inside_code_block = True
                continue
            elif line.startswith("```"):
                inside_code_block = False
                continue
            if inside_code_block:
                code_lines.append(line)

        if not code_lines:
            print("Warning: No code content detected, this task will be skipped.")
            continue

        code_content = "\n".join(code_lines).strip()

        # If the framework information is not detected, try to infer the framework from the code content
        if framework is None:
            framework = extract_framework_from_code(code_content)
            print(f"Framework inferred from the code content: {framework}")

        print(f"Selected framework: {framework}")
        print("*" * 60)
        print(f"Code implementation:\n{code_content}")
        print("*" * 60)

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

    print("Code generation for all tasks is completed. The results have been saved to output.json.")

if __name__ == "__main__":
    generate_code("input.json")
