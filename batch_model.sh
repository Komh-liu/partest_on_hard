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
    sed -i "s/\"model\": \".*\"/\"model\": \"$NEW_MODEL\"/g" "$CONFIG_FILE"
    
    # 将新的model值追加到日志文件中
    echo "Model updated to $NEW_MODEL at $(date)" >> "$LOG_FILE"
    echo "Model value updated to $NEW_MODEL"
    sh do.sh
done < "$MODELS_FILE"