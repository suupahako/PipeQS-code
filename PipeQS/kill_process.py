#!/bin/bash


pid_prefix=$1
pid_len = $2


# 查找以用户输入的4位数字开头且PID为7位数的所有进程
pids=$(ps -eo pid | grep -E "${pid_prefix}[0-9]${pid_len}")

# 检查是否找到了任何进程
if [ -z "$pids" ]; then
    echo "No processes found with PID starting with $pid_prefix."
    exit 0
fi

# 遍历找到的PID列表并杀死这些进程
for pid in $pids; do
    echo "Killing process with PID $pid"
    kill $pid
done

echo "All specified processes have been killed."
