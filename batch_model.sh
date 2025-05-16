#!/bin/bash

# 指定config.py文件的路径
CONFIG_FILE="generate/config.py"

# 指定包含新model值的文件
MODELS_FILE="models.txt"

# 指定日志文件的路径
LOG_FILE="driver/log.txt"

# 检查models.txt文件是否存在
if [ ! -f "$MODELS_FILE" ]; then
    echo "Error: $MODELS_FILE does not exist."
    exit 1
fi

# 读取models.txt文件中的每一行
while read NEW_MODEL; do
    # 使用sed命令修改model值
    sed -i "s/\"model\": \".*\"/\"model\": \"$NEW_MODEL\"/" "$CONFIG_FILE"
    
    # 检查是否修改成功
    if grep -q "\"model\": \"$NEW_MODEL\"" "$CONFIG_FILE"; then
        echo "\n Model updated to $NEW_MODEL at $(date)\n" >> "$LOG_FILE"
        echo "Model value updated to $NEW_MODEL"
    else
        echo "Failed to update model value to $NEW_MODEL" >> "$LOG_FILE"
        echo "Failed to update model value to $NEW_MODEL"
        continue
    fi

    # 运行生成脚本
    cd generate
    python generate.py
    cp output.json ../driver/output.json
    cd ..
    cd driver
    python driver.py
    cd ..
done < "$MODELS_FILE"
rm -rf /tmp/tmp*