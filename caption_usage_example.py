"""
使用caption特征的示例代码

这个文件展示了如何使用修改后的MultiModalBartForMASBA模型，
该模型现在支持加入图片caption作为额外特征。
"""

import torch
from transformers import AutoTokenizer
from bart_model import MultiModalBartForMASBA
from PIL import Image
import torchvision.transforms as transforms

# 初始化模型
def init_model_with_captions():
    """
    初始化带有caption功能的模型
    """
    # caption文件路径列表
    caption_files = [
        '/workspace/compare_model/bart/face_descriptions/twitter15_face_description_clip16.json',
        '/workspace/compare_model/bart/face_descriptions/twitter17_face_description_clip16.json'
    ]
    
    # 初始化模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MultiModalBartForMASBA(
        num_labels=3,
        text_model_name="/workspace/models/bart-large-mnli",
        language_model_type="bart",
        device=device,
        use_film=True,
        caption_files=caption_files  # 传入caption文件路径
    )
    
    model.to(device)
    return model, device


# 准备数据
def prepare_data(texts, image_paths, image_ids, model, device):
    """
    准备训练/推理数据
    
    Args:
        texts: 文本列表
        image_paths: 图片路径列表
        image_ids: 图片文件名列表 (例如 ['74960.jpg', '1739565.jpg'])
        model: 模型实例
        device: 设备
    
    Returns:
        准备好的数据字典
    """
    # 文本tokenizer
    text_tokenizer = AutoTokenizer.from_pretrained('/workspace/models/bart-large-mnli')
    # Caption tokenizer
    caption_tokenizer = AutoTokenizer.from_pretrained('/workspace/models/bert-base-uncased')
    
    # 处理文本
    text_encoding = text_tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors='pt'
    )
    
    # 获取caption文本
    caption_texts = model.get_caption_text(image_ids)
    
    # 处理caption
    caption_encoding = caption_tokenizer(
        caption_texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors='pt'
    )
    
    # 处理图像
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    images = []
    for img_path in image_paths:
        try:
            img = Image.open(img_path).convert('RGB')
            img = transform(img)
            images.append(img)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # 如果图片加载失败，使用空图像
            images.append(torch.zeros(3, 224, 224))
    
    images = torch.stack(images).to(device)
    
    # 返回所有需要的数据
    return {
        'input_ids': text_encoding['input_ids'].to(device),
        'attention_mask': text_encoding['attention_mask'].to(device),
        'caption_input_ids': caption_encoding['input_ids'].to(device),
        'caption_attention_mask': caption_encoding['attention_mask'].to(device),
        'images': images,
        'image_ids': image_ids
    }


# 训练示例
def train_example():
    """
    训练示例代码
    """
    # 初始化模型
    model, device = init_model_with_captions()
    model.train()
    
    # 示例数据
    texts = [
        "This is a sample text about sentiment",
        "Another example of text data"
    ]
    image_paths = [
        "/path/to/74960.jpg",
        "/path/to/1739565.jpg"
    ]
    image_ids = [
        "74960.jpg",
        "1739565.jpg"
    ]
    labels = torch.tensor([0, 1]).to(device)  # 示例标签
    
    # 准备数据
    batch_data = prepare_data(texts, image_paths, image_ids, model, device)
    
    # 前向传播
    outputs = model(
        input_ids=batch_data['input_ids'],
        attention_mask=batch_data['attention_mask'],
        images=batch_data['images'],
        image_ids=batch_data['image_ids'],
        caption_input_ids=batch_data['caption_input_ids'],
        caption_attention_mask=batch_data['caption_attention_mask'],
        labels=labels
    )
    
    loss = outputs['loss']
    logits = outputs['logits']
    
    print(f"Loss: {loss.item()}")
    print(f"Logits shape: {logits.shape}")
    
    # 反向传播
    # optimizer.zero_grad()
    # loss.backward()
    # optimizer.step()


# 推理示例
def inference_example():
    """
    推理示例代码
    """
    # 初始化模型
    model, device = init_model_with_captions()
    model.eval()
    
    # 示例数据
    texts = ["This is a test sentence"]
    image_paths = ["/path/to/74960.jpg"]
    image_ids = ["74960.jpg"]
    
    # 准备数据
    batch_data = prepare_data(texts, image_paths, image_ids, model, device)
    
    # 推理
    with torch.no_grad():
        outputs = model(
            input_ids=batch_data['input_ids'],
            attention_mask=batch_data['attention_mask'],
            images=batch_data['images'],
            image_ids=batch_data['image_ids'],
            caption_input_ids=batch_data['caption_input_ids'],
            caption_attention_mask=batch_data['caption_attention_mask']
        )
    
    logits = outputs['logits']
    predictions = torch.argmax(logits, dim=-1)
    
    print(f"Predictions: {predictions}")
    print(f"Logits: {logits}")


if __name__ == "__main__":
    print("=" * 50)
    print("训练示例")
    print("=" * 50)
    # train_example()  # 取消注释以运行训练示例
    
    print("\n" + "=" * 50)
    print("推理示例")
    print("=" * 50)
    # inference_example()  # 取消注释以运行推理示例
    
    print("\n使用说明:")
    print("1. 在初始化模型时，通过caption_files参数传入caption JSON文件路径列表")
    print("2. 准备数据时，需要提供image_ids（图片文件名列表）")
    print("3. 模型会自动根据image_ids从加载的caption数据中获取对应的caption文本")
    print("4. caption文本会被编码并与其他特征（文本、图像）一起融合")
    print("5. 如果某个图片没有对应的caption，会使用空字符串作为默认值")

