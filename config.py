import torch

class Config:
    def __init__(self, dataset_name="twitter2015", parent_output_dir=None):
        # 设备设置
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 多GPU设置
        self.use_multi_gpu = False  # 是否使用多GPU
        self.gpu_ids = [0, 1, 2]  # 使用的GPU ID列表
        self.main_gpu = 0  # 主GPU
        
        # 数据路径
        self.data_root = "/workspace/datasets/IJCAI2019_data"
        self.dataset_name = dataset_name
        self.train_path = f"{self.data_root}/{self.dataset_name}/train.tsv"
        self.val_path = f"{self.data_root}/{self.dataset_name}/dev.tsv"
        self.test_path = f"{self.data_root}/{self.dataset_name}/test.tsv"
        # 图像路径
        self.train_image_dir = f"{self.data_root}/{self.dataset_name}_images"
        self.val_image_dir = f"{self.data_root}/{self.dataset_name}_images"
        self.test_image_dir = f"{self.data_root}/{self.dataset_name}_images"
        
        # 模型参数
        # language_model_type 可选: "bart" 或 "bert"
        self.language_model_type = "bart"
        # 根据 language_model_type 指定对应的模型路径
        self.model_name = "/workspace/models/bart-large-mnli" if self.language_model_type == "bart" else "/workspace/models/bert-base-uncased"
        self.num_labels = 3  # negative(0), neutral(1), positive(2)
        
        # 训练参数
        self.num_epochs = 20
        self.batch_size = 16
        self.learning_rate = 2e-5
        self.warmup_steps = 0
        self.max_grad_norm = 1.0
        self.max_length = 128
        
        # 输出路径
        if parent_output_dir is not None:
            # 如果指定了父文件夹，则在父文件夹下创建数据集子文件夹
            # 提取年份，如 twitter2015 -> 2015
            year = dataset_name.replace("twitter", "")
            self.output_dir = f"{parent_output_dir}/{year}"
        else:
            # 默认行为：独立的输出文件夹
            import time
            self.output_dir = f"checkpoints/{self.language_model_type}_{self.dataset_name}_{time.strftime('%Y%m%d_%H%M%S')}" 