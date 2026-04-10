#!/bin/bash

# 快速开始脚本 - 测试和训练单个任务

echo "================================================"
echo "多任务模型快速开始指南"
echo "================================================"

# 1. 测试模型和数据加载
echo ""
echo "步骤 1: 测试模型和数据加载"
echo "------------------------------------------------"
echo "运行测试脚本验证所有组件是否正常工作..."
python test_multitask.py

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ 测试失败！请检查错误信息并修复问题。"
    exit 1
fi

echo ""
echo "✓ 测试通过！"

# 2. 选择任务和数据集
echo ""
echo "================================================"
echo "步骤 2: 选择要训练的任务"
echo "================================================"
echo ""
echo "请选择任务类型:"
echo "  1) MABSC  - Multimodal Aspect-Based Sentiment Classification"
echo "  2) MATE   - Multimodal Aspect Term Extraction"
echo "  3) JMABSC - Joint Multimodal ABSC"
echo ""
read -p "请输入选项 (1-3): " task_choice

case $task_choice in
    1)
        TASK="mabsc"
        ;;
    2)
        TASK="mate"
        ;;
    3)
        TASK="joint"
        ;;
    *)
        echo "无效选项，使用默认任务: mabsc"
        TASK="mabsc"
        ;;
esac

echo ""
echo "请选择数据集:"
echo "  1) Twitter2015"
echo "  2) Twitter2017"
echo ""
read -p "请输入选项 (1-2): " dataset_choice

case $dataset_choice in
    1)
        DATASET="twitter2015"
        ;;
    2)
        DATASET="twitter2017"
        ;;
    *)
        echo "无效选项，使用默认数据集: twitter2015"
        DATASET="twitter2015"
        ;;
esac

# 3. 设置训练参数
echo ""
echo "================================================"
echo "步骤 3: 设置训练参数"
echo "================================================"
echo ""
echo "使用默认参数还是自定义?"
echo "  1) 使用默认参数 (推荐)"
echo "  2) 自定义参数"
echo ""
read -p "请输入选项 (1-2): " param_choice

if [ "$param_choice" == "2" ]; then
    read -p "批次大小 (默认: 16): " BATCH_SIZE
    BATCH_SIZE=${BATCH_SIZE:-16}
    
    read -p "训练轮数 (默认: 10): " EPOCHS
    EPOCHS=${EPOCHS:-10}
    
    read -p "学习率 (默认: 2e-5): " LR
    LR=${LR:-2e-5}
else
    BATCH_SIZE=16
    EPOCHS=10
    LR=2e-5
fi

# 4. 显示配置并确认
echo ""
echo "================================================"
echo "训练配置"
echo "================================================"
echo "任务类型: $TASK"
echo "数据集: $DATASET"
echo "批次大小: $BATCH_SIZE"
echo "训练轮数: $EPOCHS"
echo "学习率: $LR"
echo "输出目录: checkpoints/${TASK}_${DATASET}"
echo "================================================"
echo ""
read -p "确认开始训练? (y/n): " confirm

if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
    echo "训练已取消"
    exit 0
fi

# 5. 开始训练
echo ""
echo "================================================"
echo "开始训练..."
echo "================================================"

python train_multitask.py \
    --task $TASK \
    --dataset $DATASET \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --learning_rate $LR \
    --use_image \
    --use_vad \
    --output_dir checkpoints/${TASK}_${DATASET}

if [ $? -eq 0 ]; then
    echo ""
    echo "================================================"
    echo "🎉 训练完成!"
    echo "================================================"
    echo "模型保存在: checkpoints/${TASK}_${DATASET}/best_model.pth"
    echo ""
    echo "查看训练配置: checkpoints/${TASK}_${DATASET}/config.json"
    echo "================================================"
else
    echo ""
    echo "================================================"
    echo "❌ 训练失败，请检查错误信息"
    echo "================================================"
    exit 1
fi

