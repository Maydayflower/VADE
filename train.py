import torch
from transformers import BartTokenizer, BertTokenizer, get_linear_schedule_with_warmup
from bart_model import MultiModalBartForMASBA
from utils.data_processor import load_data, compute_metrics
from visualizer import BartVisualizer
import numpy as np
from tqdm import tqdm
import logging
import os
import json
import shutil
from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def snapshot_related_py_files(output_dir):
    """
    将当前目录下的bart_model.py, train.py, config.py都复制到output_dir下，用带快照时间戳的文件名，避免重复复制。
    """
    import time
    snapshot_paths = {}
    files_to_snapshot = ["bart_model.py", "train.py", "config.py"]
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    for fname in files_to_snapshot:
        src_path = os.path.join(os.path.dirname(__file__), fname)
        dst_path = os.path.join(output_dir, f'{fname.replace(".py","")}_snapshot_{timestamp}.py')
        try:
            shutil.copyfile(src_path, dst_path)
            logger.info(f'{fname} 快照已保存到 {dst_path}')
            snapshot_paths[fname] = dst_path
        except Exception as e:
            logger.warning(f'保存{fname}快照失败: {e}')
            snapshot_paths[fname] = None
    return snapshot_paths

def train(config):
    # 使用config中的设备设置
    device = config.device
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    logger.info(f"使用设备: {device}")
    
    # 多GPU设置
    if config.use_multi_gpu and torch.cuda.device_count() > 1:
        logger.info(f"使用多GPU训练: {torch.cuda.device_count()} GPUs 可用")
        logger.info(f"将使用GPU IDs: {config.gpu_ids}")
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, config.gpu_ids))
    else:
        logger.info(f"使用单GPU训练")
    
    # 加载tokenizer和模型
    if config.language_model_type == 'bert':
        text_tokenizer = BertTokenizer.from_pretrained(config.model_name)
    else:
        text_tokenizer = BartTokenizer.from_pretrained(config.model_name)
    vad_tokenizer = BertTokenizer.from_pretrained('/workspace/models/bert-base-uncased')

    logger.info("初始化MultiModalBartForMASBA模型...")
    model = MultiModalBartForMASBA(
        num_labels=config.num_labels,
        text_model_name=config.model_name,
        language_model_type=config.language_model_type,
        device=device
    )
    model.to(device)
    
    # 多GPU并行
    if config.use_multi_gpu and torch.cuda.device_count() > 1:
        logger.info("包装模型为DataParallel...")
        model = torch.nn.DataParallel(model, device_ids=config.gpu_ids, output_device=config.main_gpu)
        logger.info(f"✓ 模型已在 {len(config.gpu_ids)} 个GPU上并行")
    
    # 统计参数
    model_for_params = model.module if hasattr(model, 'module') else model
    total_params = sum(p.numel() for p in model_for_params.parameters())
    trainable_params = sum(p.numel() for p in model_for_params.parameters() if p.requires_grad)
    logger.info(f"总参数: {total_params:,}")
    logger.info(f"可训练参数: {trainable_params:,}")
    
    # 加载数据
    train_dataloader = load_data(
        config.train_path, 
        config.train_image_dir, 
        text_tokenizer,
        vad_tokenizer,
        config.batch_size
    )
    val_dataloader = load_data(
        config.val_path,
        config.val_image_dir,
        text_tokenizer,
        vad_tokenizer,
        config.batch_size
    )
    test_dataloader = load_data(
        config.test_path,
        config.test_image_dir,
        text_tokenizer,
        vad_tokenizer,
        config.batch_size
    )
    # 优化器设置
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    total_steps = len(train_dataloader) * config.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=total_steps
    )
    
    # 训练循环
    best_accuracy_test = 0
    best_val_metrics = None
    best_test_metrics = None
    best_epoch = 0
    
    # 用于记录训练历史
    training_history = {
        'epochs': [],
        'train_loss': [],
        'val_accuracy': [],
        'val_f1_macro': [],
        'val_f1_weighted': [],
        'test_accuracy': [],
        'test_f1_macro': [],
        'test_f1_weighted': []
    }

    snapshot_paths = None
    snapshot_done = False

    for epoch in range(config.num_epochs):
        model.train()
        total_loss = 0
        
        for batch in tqdm(train_dataloader, desc=f'Epoch {epoch + 1}'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            vad_input_ids = batch['vad_input_ids'].to(device)
            vad_attention_mask = batch['vad_attention_mask'].to(device)
            images = batch['images'].to(device)
            image_path = batch['image_path']
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask, vad_input_ids, vad_attention_mask, images, image_path, labels=labels)
            loss = outputs['loss']
            
            # 在DataParallel模式下，loss是向量，需要平均
            if loss.dim() > 0:
                loss = loss.mean()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            
        avg_train_loss = total_loss / len(train_dataloader)
        logger.info(f'Average training loss: {avg_train_loss}')
        
        # 每个epoch都在验证集与测试集评估
        val_metrics = evaluate(model, val_dataloader, device)
        test_metrics = evaluate(model, test_dataloader, device)
        logger.info(f'Epoch {epoch+1} Validation metrics: {val_metrics}')
        logger.info(f'Epoch {epoch+1} Test metrics: {test_metrics}')
        
        # 记录训练历史
        training_history['epochs'].append(epoch + 1)
        training_history['train_loss'].append(avg_train_loss)
        training_history['val_accuracy'].append(val_metrics['accuracy'])
        training_history['val_f1_macro'].append(val_metrics['f1_macro'])
        training_history['val_f1_weighted'].append(val_metrics['f1_weighted'])
        training_history['test_accuracy'].append(test_metrics['accuracy'])
        training_history['test_f1_macro'].append(test_metrics['f1_macro'])
        training_history['test_f1_weighted'].append(test_metrics['f1_weighted'])

        # 保存test集acc最好的模型
        if test_metrics['accuracy'] > best_accuracy_test:
            best_accuracy_test = test_metrics['accuracy']
            best_val_metrics = val_metrics
            best_test_metrics = test_metrics
            best_epoch = epoch + 1
            if not os.path.exists(config.output_dir):
                os.makedirs(config.output_dir)
                model_info = model.model_info
                with open(os.path.join(config.output_dir, 'model_info.json'), 'w', encoding='utf-8') as f:
                    json.dump(model_info, f, ensure_ascii=False, indent=4)
            # 保存模型时，如果是DataParallel，保存module
            model_to_save = model.module if hasattr(model, 'module') else model
            torch.save(model_to_save.state_dict(), os.path.join(config.output_dir, 'best_model.pt'))
            # 只在第一次保存时快照train.py, config.py, bart_model.py
            if not snapshot_done:
                snapshot_paths = snapshot_related_py_files(config.output_dir)
                snapshot_done = True
            logger.info(f'New best model on test set is saved (accuracy={test_metrics["accuracy"]:.4f})')
            
            # 保存指标（包含每个类别的详细指标）
            metrics_to_save = {
                "val_metrics": {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in best_val_metrics.items()},
                "test_metrics": {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in best_test_metrics.items()},
                "epoch": epoch + 1,
                "bart_model_snapshot": snapshot_paths["bart_model.py"] if snapshot_paths and "bart_model.py" in snapshot_paths else "",
                "train_py_snapshot": snapshot_paths["train.py"] if snapshot_paths and "train.py" in snapshot_paths else "",
                "config_py_snapshot": snapshot_paths["config.py"] if snapshot_paths and "config.py" in snapshot_paths else ""
            }
            with open(os.path.join(config.output_dir, 'best_model_metrics.json'), 'w', encoding='utf-8') as f:
                json.dump(metrics_to_save, f, ensure_ascii=False, indent=4)
    
    # 训练完成后，使用最佳模型生成可视化
    logger.info("\n" + "="*80)
    logger.info("训练完成！开始生成可视化结果...")
    logger.info("="*80)
    
    # 加载最佳模型
    model_to_load = model.module if hasattr(model, 'module') else model
    model_to_load.load_state_dict(torch.load(os.path.join(config.output_dir, 'best_model.pt')))
    
    # 获取预测结果
    logger.info("在训练集上评估并生成可视化...")
    train_metrics, train_preds, train_labels = evaluate(model, train_dataloader, device, return_predictions=True)
    
    logger.info("在验证集上评估并生成可视化...")
    val_metrics, val_preds, val_labels = evaluate(model, val_dataloader, device, return_predictions=True)
    
    logger.info("在测试集上评估并生成可视化...")
    test_metrics, test_preds, test_labels = evaluate(model, test_dataloader, device, return_predictions=True)
    
    # 创建可视化器
    visualizer = BartVisualizer(output_dir=config.output_dir)
    
    # 生成训练集可视化
    visualizer.visualize_all(train_labels, train_preds, train_metrics, "Train")
    
    # 生成验证集可视化
    visualizer.visualize_all(val_labels, val_preds, val_metrics, "Validation")
    
    # 生成测试集可视化
    visualizer.visualize_all(test_labels, test_preds, test_metrics, "Test")
    
    # 生成指标对比图
    logger.info("生成指标对比图...")
    visualizer.plot_metrics_comparison(train_metrics, val_metrics, test_metrics)
    
    # 生成训练历史图
    logger.info("生成训练历史图...")
    visualizer.plot_training_history(training_history)
    
    # 保存训练历史
    history_path = os.path.join(config.output_dir, 'training_history.json')
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(training_history, f, ensure_ascii=False, indent=4)
    logger.info(f"训练历史已保存: {history_path}")
    
    logger.info("\n" + "="*80)
    logger.info(f"所有可视化结果已保存到: {config.output_dir}")
    logger.info(f"最佳模型 - Epoch {best_epoch}")
    logger.info(f"测试集 Accuracy: {best_accuracy_test:.4f}")
    logger.info("="*80 + "\n")
            
def evaluate(model, dataloader, device, return_predictions=False):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            vad_input_ids = batch['vad_input_ids'].to(device)
            vad_attention_mask = batch['vad_attention_mask'].to(device)
            images = batch['images'].to(device)
            image_path = batch['image_path']  # 添加image_path
            labels = batch['labels']
            
            outputs = model(input_ids, attention_mask, vad_input_ids, vad_attention_mask, images, image_path)
            if isinstance(outputs, dict):
                logits = outputs['logits']
            else:
                logits = outputs
            
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    
    metrics = compute_metrics(all_preds, all_labels, return_numpy='True')
    
    if return_predictions:
        return metrics, np.array(all_preds), np.array(all_labels)
    return metrics

if __name__ == '__main__':
    import time
    import argparse
    
    parser = argparse.ArgumentParser(description="训练MultiModalBartForMASBA模型")
    parser.add_argument('--use_multi_gpu', action='store_true',
                        help='是否使用多GPU训练')
    parser.add_argument('--gpu_ids', type=str, default='0,1,2',
                        help='使用的GPU ID，逗号分隔（例如：0,1,2）')
    parser.add_argument('--datasets', type=str, default='twitter2015,twitter2017',
                        help='要训练的数据集，逗号分隔')
    
    args = parser.parse_args()
    
    # 创建父文件夹（带时间戳）
    parent_dir = f"checkpoints/bart_both_datasets_{time.strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(parent_dir, exist_ok=True)
    logger.info(f"创建父输出文件夹: {parent_dir}")
    
    # 要训练的数据集列表
    datasets = args.datasets.split(',')
    
    logger.info(f"\n配置信息:")
    logger.info(f"  数据集: {datasets}")
    logger.info(f"  多GPU训练: {args.use_multi_gpu}")
    if args.use_multi_gpu:
        logger.info(f"  GPU IDs: {args.gpu_ids}")
    logger.info("")
    
    # 依次训练每个数据集
    for dataset_name in datasets:
        logger.info(f"\n{'='*80}")
        logger.info(f"开始训练数据集: {dataset_name}")
        logger.info(f"{'='*80}\n")
        
        # 为每个数据集创建配置
        config = Config(dataset_name=dataset_name, parent_output_dir=parent_dir)
        
        # 设置多GPU配置
        config.use_multi_gpu = args.use_multi_gpu
        config.gpu_ids = [int(gpu_id) for gpu_id in args.gpu_ids.split(',')]
        
        logger.info(f"输出路径: {config.output_dir}")
        
        # 训练模型
        train(config)
        
        logger.info(f"\n{'='*80}")
        logger.info(f"数据集 {dataset_name} 训练完成")
        logger.info(f"{'='*80}\n")
    
    logger.info(f"\n所有数据集训练完成！结果保存在: {parent_dir}") 