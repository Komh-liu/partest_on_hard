import os
import json
import shutil

# 指定config.py文件的路径
CONFIG_FILE = "generate/config.py"
# 指定包含新model值的文件
MODELS_FILE = "models.txt"
# 指定日志文件的路径
LOG_FILE = "driver/log.txt"

# 检查models.txt文件是否存在
if not os.path.exists(MODELS_FILE):
    print(f"Error: {MODELS_FILE} does not exist.")
    exit(1)

# 读取models.txt文件中的每一行
with open(MODELS_FILE, 'r') as file:
    new_models = file.readlines()

# 读取并解析config.py文件
with open(CONFIG_FILE, 'r') as file:
    config_content = file.read()

# 使用正则表达式找到model值
import re
config_match = re.search(r'"model":\s*"(.*)"', config_content)
if config_match:
    original_model = config_match.group(1)
else:
    print("Error: 'model' key not found in config.py.")
    exit(1)

for new_model in new_models:
    new_model = new_model.strip()  # 去除可能的换行符
    # 使用字符串替换来更新model值
    config_content = re.sub(r'"model":\s*".*"', f'"model": "{new_model}"', config_content)

    # 将新的config内容写回config.py文件
    with open(CONFIG_FILE, 'w') as file:
        file.write(config_content)
    
    # 将新的model值追加到日志文件中
    with open(LOG_FILE, 'a') as log_file:
        log_file.write(f"\nModel updated to {new_model} at {os.path.getmtime(CONFIG_FILE)}\n")
    print(f"Model value updated to {new_model}")

    # 运行生成脚本
    os.chdir("generate")
    os.system("python generate.py")
    shutil.copy("output.json", "../driver/output.json")
    os.chdir("..")
    os.chdir("driver")
    os.system("python driver.py")
    os.chdir("..")

print("All models have been updated.")