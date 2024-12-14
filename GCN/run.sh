#!/bin/bash
#SBATCH --job-name=python-job       # 作业名称
#SBATCH --output=output.log         # 输出文件
#SBATCH --error=error.log           # 错误日志
#SBATCH --time=08:00:00             # 运行时间限制
#SBATCH --mem=64GB                   # 内存要求
#SBATCH --gres=gpu:1                  # 申请 1 个 GPU
#SBATCH --ntasks=1                  # 任务数

cd ~/GNN_E
conda activate GNN_E_clone             # 加载 Python 环境（如果需要）
python GCN/GCN.py               # 运行 Python 脚本