import torch
import torch.nn as nn
from transformers import BartModel, BertModel, ViTModel, BartTokenizer, CLIPProcessor, CLIPModel
import torch.nn.functional as F
import math
import sys
import os
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), 'vad_utils'))
from vad_encoder import Transformer as VADEncoder

# 添加CLIP VAD predictor的路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'clip'))
from stage1_alignment.train_text_vad import ClipImageTextAlignmentModel
from stage2_downstream.train_vad_downstream import ImageFeatureVADModel

#################################################################################################
class VADPredictor(nn.Module):
    def __init__(self, bert_model_name='/workspace/models/bert-base-uncased'):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        print(f"[VADPredictor] Using text model: {bert_model_name}")
        self.regressor = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 3)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_hidden = outputs.last_hidden_state[:, 0, :]
        return self.regressor(cls_hidden)


class MultiModalBartForMASBA(nn.Module):
    def __init__(
        self, 
        num_labels=3, 
        text_model_name="/workspace/models/bart-large-mnli", 
        language_model_type="bart", 
        device=torch.device('cuda'), 
        use_film=True, 
        image_encoder_type="clip",  # "vit" or "clip"
        image_encoder_path=None    # None, or image encoder path (for custom CLIP)
    ):
        super().__init__()
        self.model_info = {
            "text_model_type": language_model_type,
            "text_model_name": text_model_name,
            "vad_predictor_path": "/workspace/compare_model/bart/clip/vad_predictor.pt",
            "image_encoder_type": image_encoder_type,
            "image_encoder_path": image_encoder_path,
            "image_out_dim": None,
        }
        if image_encoder_type == "vit":
            self.model_info["image_encoder_path"] = "/workspace/models/vit-base-patch16-224"
            self.model_info["image_out_dim"] = 768
        elif image_encoder_type == "clip":
            self.model_info["image_encoder_path"] = "/workspace/models/clip-vit-base-patch32"
            self.model_info["image_out_dim"] = 512
        self.model_info["num_labels"] = num_labels
        self.device = device  # 保存 device 以便后续使用
        # 文本编码器
        self.language_model_type = language_model_type
        if self.language_model_type == "bert":
            self.text_model = BertModel.from_pretrained(text_model_name)
            self.text_dim = self.text_model.config.hidden_size
            print(f"[MultiModalBartForMASBA] Using text model: BertModel, checkpoint: {text_model_name}")
        else:
            self.text_model = BartModel.from_pretrained(text_model_name)
            self.text_dim = self.text_model.config.hidden_size
            print(f"[MultiModalBartForMASBA] Using text model: BartModel, checkpoint: {text_model_name}")
        self.vad_model = VADPredictor().to(device)
        self.vad_model.load_state_dict(torch.load("/workspace/compare_model/bart/clip/vad_predictor.pt", map_location=device))
        self.model_info["vad_predictor_path"] = "/workspace/compare_model/bart/clip/vad_predictor.pt"
        # ---------------------- 图像编码器选择 --------------
        self.image_encoder_type = image_encoder_type
        if self.image_encoder_type == "vit":
            self.image_encoder = ViTModel.from_pretrained('/workspace/models/vit-base-patch16-224')
            self.model_info["image_encoder_path"] = "/workspace/models/vit-base-patch16-224"
            print(f"[MultiModalBartForMASBA] Using vision model: ViTModel, checkpoint: /workspace/models/vit-base-patch16-224")
            self.processor = CLIPProcessor.from_pretrained("/workspace/models/clip-vit-base-patch32")
        elif self.image_encoder_type == "clip":
            # CLIP: 'openai/clip-vit-base-patch16' or a local path
            clip_path = image_encoder_path or "/workspace/models/clip-vit-base-patch32"
            self.model_info["image_encoder_path"] = clip_path
            self.image_encoder = CLIPModel.from_pretrained(clip_path)
            print(f"[MultiModalBartForMASBA] Using vision model: CLIPModel, checkpoint: {clip_path}")
            # 获取 CLIP 图像特征的输出维度
            try:
                if hasattr(self.image_encoder, 'vision_model'):
                    vision_config = self.image_encoder.vision_model.config
                    if hasattr(self.image_encoder, 'visual_projection'):
                        self.image_out_dim = self.image_encoder.visual_projection.out_features
                    elif hasattr(self.image_encoder.config, 'projection_dim'):
                        self.image_out_dim = self.image_encoder.config.projection_dim
                    elif hasattr(vision_config, 'projection_dim'):
                        self.image_out_dim = vision_config.projection_dim
                    elif hasattr(vision_config, 'hidden_size'):
                        self.image_out_dim = vision_config.hidden_size
                    else:
                        raise AttributeError("Cannot determine vision output dim from config")
                elif hasattr(self.image_encoder, 'visual'):
                    if hasattr(self.image_encoder.visual, 'projection'):
                        self.image_out_dim = self.image_encoder.visual.projection.out_features
                    else:
                        raise AttributeError("Cannot find visual.projection")
                else:
                    raise AttributeError("Cannot find vision_model or visual")
            except (AttributeError, Exception) as e:
                try:
                    dummy_input = torch.zeros(1, 3, 224, 224)
                    self.image_encoder.eval()
                    with torch.no_grad():
                        dummy_output = self.image_encoder.get_image_features(pixel_values=dummy_input)
                        self.image_out_dim = dummy_output.shape[-1]
                    del dummy_input, dummy_output
                except Exception as e2:
                    print(f"Warning: Could not determine CLIP image output dimension from config, using default 512. Error: {e}")
                    self.image_out_dim = 512
            self.processor = CLIPProcessor.from_pretrained(clip_path)
        else:
            raise ValueError(f"Unknown image_encoder_type: {self.image_encoder_type}")
        
        # 图像特征投影层：将编码器输出维度映射到text_dim
        self.image_projection = nn.Linear(self.image_out_dim, self.text_dim)

        # VAD三维映射为卷积核参数
        self.vad_to_conv_weight = nn.Sequential(
            nn.Linear(3, self.text_dim),
            # nn.ReLU(),
            # nn.Linear(128, self.text_dim)
        )
        self.vad_to_conv_bias = nn.Sequential(
            nn.Linear(3, self.text_dim),
            # nn.ReLU(),
            # nn.Linear(128, self.text_dim)
        )

        self.image_vad_projection = nn.Sequential(
            nn.Linear(self.text_dim+3, self.text_dim),
            # nn.LayerNorm(self.text_dim),
            # nn.GELU(),
            # nn.Dropout(0.1),
            # nn.Linear(self.text_dim, self.text_dim),
            # nn.LayerNorm(self.text_dim),
            # nn.Dropout(0.1)
        )
        
        self.text_vad_projection = nn.Sequential(
            nn.Linear(self.text_dim+3, self.text_dim),
        )

        # 兼容CLIP VAD链
        self.clip_alignment_model_path = "/workspace/models/clip-vit-base-patch32"
        self.clip_model = ClipImageTextAlignmentModel(self.clip_alignment_model_path)
        self.clip_model.load_state_dict(torch.load("/workspace/compare_model/bart/clip/output/clip_alignment_model/best_clip_alignment.pt", map_location=device))
        self.image_vad_predictor = ImageFeatureVADModel(input_dim=512, hidden_dim=256)
        self.image_vad_predictor.load_state_dict(torch.load("/workspace/compare_model/bart/clip/output/vad_downstream_model/best_vad_downstream.pt", map_location=device))
        self.image_encoder = CLIPModel.from_pretrained(self.clip_alignment_model_path)

        self.processor = CLIPProcessor.from_pretrained(self.clip_alignment_model_path)
        self.fusion = nn.Linear(self.text_dim * 2, self.text_dim)
        self.classifier = nn.Linear(self.text_dim, num_labels)
        self.combine_classifier = nn.Linear(self.text_dim * 2 + 3, num_labels)
        self.text_classifier = nn.Linear(self.text_dim, num_labels)

    def forward(self, input_ids, attention_mask, vad_input_ids=None, vad_attention_mask=None, images=None, image_path=None, target_mask=None, labels=None):
        # 文本特征提取
        if vad_input_ids is None:
            vad_input_ids = input_ids
        if vad_attention_mask is None:
            vad_attention_mask = attention_mask
        vad_features = self.vad_model(vad_input_ids, vad_attention_mask)  # (batch, 3)
        
        # 图像vad提取
        current_batch_size = images.size(0) if images is not None else input_ids.size(0)
        current_image_paths = image_path[:current_batch_size] if isinstance(image_path, (list, tuple)) else image_path
        
        # 获取视觉输入
        if images is not None and self.image_encoder_type == "vit":
            # 加载图像文件
            from PIL import Image
            loaded_images = [Image.open(p).convert("RGB") for p in current_image_paths]
            vision_inputs = self.processor(images=loaded_images, return_tensors="pt")
            pixel_values = vision_inputs['pixel_values'].to(images.device)
            with torch.no_grad():
                clip_image_features = self.clip_model.get_image_features(pixel_values=pixel_values)
                image_vad_output = self.image_vad_predictor(clip_image_features)
                image_vad_preds = image_vad_output # (batch, 3)
        elif images is not None and self.image_encoder_type == "clip":
            # 加载图像文件
            from PIL import Image
            loaded_images = [Image.open(p).convert("RGB") for p in current_image_paths]
            vision_inputs = self.processor(images=loaded_images, return_tensors="pt")
            pixel_values = vision_inputs['pixel_values'].to(images.device)
            with torch.no_grad():
                image_feat = self.image_encoder.get_image_features(pixel_values=pixel_values)
                clip_image_features = self.clip_model.get_image_features(pixel_values=pixel_values)
                image_vad_output = self.image_vad_predictor(clip_image_features)
                image_vad_preds = image_vad_output
        else:
            image_vad_preds = None
            image_feat = None

        outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        text_hidden = outputs.last_hidden_state

        conv_weight = self.vad_to_conv_weight(vad_features)  # (batch, text_dim)
        conv_bias = self.vad_to_conv_bias(vad_features)      # (batch, text_dim)
        text_hidden_t = text_hidden.transpose(1, 2)  # (batch, text_dim, seq_len)
        conv_weight = conv_weight.unsqueeze(-1)  # (batch, text_dim, 1)
        conv_bias = conv_bias.unsqueeze(-1)      # (batch, text_dim, 1)
        conv_out = text_hidden_t * conv_weight + conv_bias  # (batch, text_dim, seq_len)
        conv_out_pooled = torch.mean(conv_out, dim=2)  # (batch, text_dim)

        # 图像特征提取
        if images is not None:
            if self.image_encoder_type == "vit":
                image_outputs = self.image_encoder(images)
                image_features = image_outputs.last_hidden_state[:, 0, :]  # (batch, 768)
            elif self.image_encoder_type == "clip":
                image_features = self.image_encoder.get_image_features(pixel_values=pixel_values)
            else:
                raise ValueError(f"Unknown image_encoder_type: {self.image_encoder_type}")

            image_features = self.image_projection(image_features)  # (batch, text_dim)
            image_features = torch.cat([image_features, image_vad_preds], dim=1)  # (batch, text_dim+3)
        else:
            image_features = torch.zeros((text_hidden.size(0), self.text_dim+ 3), device=text_hidden.device, dtype=text_hidden.dtype)
        
        
        # 融合卷积后的文本特征和图像特征
        fused_features = torch.cat([conv_out_pooled, image_features], dim=1)
        
        # 分类
        logits = self.combine_classifier(fused_features)
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)

        return {
            'loss': loss,
            'logits': logits,
            'features': fused_features
        } if loss is not None else logits



#################################################################################################
# 图像VAD融合模型
#################################################################################################

class ImageVADPredictor(nn.Module):
    """
    图像VAD预测器 - 使用预训练的CLIP VAD模型
    
    加载训练好的CLIP图像特征提取器和VAD预测头
    """
    def __init__(
        self,
        clip_checkpoint_path: str,
        vad_checkpoint_path: str,
        clip_model_path: str = "/workspace/models/clip-vit-base-patch32",
        hidden_dim: int = 256,
        dropout: float = 0.1,
        freeze: bool = True
    ):
        super().__init__()
        
        # 加载CLIP模型
        self.clip_model = ClipImageTextAlignmentModel(clip_model_path)
        print(f"[ImageVADPredictor] Using vision model: {clip_model_path}")
        if os.path.exists(clip_checkpoint_path):
            self.clip_model.load_state_dict(torch.load(clip_checkpoint_path, map_location='cpu'))
            print(f"[ImageVADPredictor] Loaded CLIP model from {clip_checkpoint_path}")
        
        # VAD预测头 (简单MLP) - 结构与ImageFeatureVADModel保持一致
        # 注意：这里使用self.mlp来匹配保存的checkpoint结构
        self.mlp = nn.Sequential(
            nn.Linear(512, hidden_dim),  # CLIP features: 512-dim
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 3),  # VAD: [V, A, D]
        )
        
        # 加载VAD预测头的权重
        if os.path.exists(vad_checkpoint_path):
            vad_state = torch.load(vad_checkpoint_path, map_location='cpu')
            mlp_state = {k: v for k, v in vad_state.items() if k.startswith('mlp.')}
            if mlp_state:
                new_state = {}
                for k, v in mlp_state.items():
                    new_key = k.replace('mlp.', '')
                    new_state[new_key] = v
                self.mlp.load_state_dict(new_state)
                print(f"[ImageVADPredictor] Loaded VAD predictor from {vad_checkpoint_path}")
            else:
                try:
                    self.mlp.load_state_dict(vad_state)
                    print(f"[ImageVADPredictor] Loaded VAD predictor from {vad_checkpoint_path}")
                except:
                    print(f"[ImageVADPredictor] Warning: Could not load VAD predictor weights")
        
        # 是否冻结
        if freeze:
            for param in self.parameters():
                param.requires_grad = False
            print("[ImageVADPredictor] Model frozen")
    
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        with torch.no_grad() if not self.training else torch.enable_grad():
            image_features = self.clip_model.get_image_features(pixel_values=pixel_values)
        vad = self.mlp(image_features)
        return vad


class BartWithImageVAD(nn.Module):
    """
    BART + 图像VAD融合模型
    
    工作流程：
    1. 文本通过BART编码器 -> 文本特征
    2. 图像通过CLIP VAD预测器 -> 图像VAD向量 (3维)
    3. 图像VAD通过MLP映射到高维空间
    4. 融合文本特征和图像VAD特征
    5. 分类预测
    """
    
    def __init__(
        self,
        num_labels: int = 3,
        bart_model_name: str = "/workspace/models/bart-large-mnli",
        clip_checkpoint: str = "/workspace/compare_model/bart/clip/output/clip_alignment_model/best_clip_alignment.pt",
        vad_checkpoint: str = "/workspace/compare_model/bart/clip/output/vad_downstream_model/best_vad_downstream.pt",
        clip_model_path: str = "/workspace/models/clip-vit-base-patch32",
        freeze_bart: bool = False,
        freeze_image_vad: bool = True,
        vad_fusion_mode: str = "concat",  # concat, add, film, gate
        use_text_vad: bool = True
    ):
        super().__init__()
        
        # 1. BART编码器
        self.bart = BartModel.from_pretrained(bart_model_name)
        self.hidden_dim = self.bart.config.d_model  # 1024 for bart-large
        print(f"[BartWithImageVAD] Using text model: BartModel, checkpoint: {bart_model_name}")
        if freeze_bart:
            for param in self.bart.parameters():
                param.requires_grad = False
            print("[BartWithImageVAD] BART frozen")
        
        # 2. 图像VAD预测器
        self.image_vad_predictor = ImageVADPredictor(
            clip_checkpoint_path=clip_checkpoint,
            vad_checkpoint_path=vad_checkpoint,
            clip_model_path=clip_model_path,
            freeze=freeze_image_vad
        )
        
        # 3. 可选：文本VAD模型
        self.use_text_vad = use_text_vad
        if use_text_vad:
            self.text_vad_model = VADModel()
            text_vad_ckpt = "/workspace/compare_model/dimabsa/best_bert_anew_model.pth"
            print(f"[BartWithImageVAD] Using text VAD model: /workspace/compare_model/dimabsa/best_bert_anew_model.pth")
            if os.path.exists(text_vad_ckpt):
                self.text_vad_model.load_state_dict(torch.load(text_vad_ckpt, map_location='cpu'))
                print(f"[BartWithImageVAD] Loaded text VAD model from {text_vad_ckpt}")
            for param in self.text_vad_model.parameters():
                param.requires_grad = False
        
        # 4. VAD特征处理
        self.vad_fusion_mode = vad_fusion_mode
        
        # VAD投影层：将3维VAD映射到hidden_dim
        self.image_vad_projection = nn.Sequential(
            nn.Linear(3, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.Dropout(0.1)
        )
        
        if use_text_vad:
            self.text_vad_projection = nn.Sequential(
                nn.Linear(3, 256),
                nn.LayerNorm(256),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(256, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                nn.Dropout(0.1)
            )
        
        # 5. 特征融合层
        if vad_fusion_mode == "concat":
            fusion_input_dim = self.hidden_dim * (3 if use_text_vad else 2)
            self.fusion_layer = nn.Sequential(
                nn.Linear(fusion_input_dim, self.hidden_dim * 2),
                nn.LayerNorm(self.hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(self.hidden_dim * 2, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                nn.Dropout(0.1)
            )
        elif vad_fusion_mode == "add":
            pass
        elif vad_fusion_mode == "film":
            self.image_vad_film = FiLMCondition(in_dim=3, embed_dim=self.hidden_dim)
            if use_text_vad:
                self.text_vad_film = FiLMCondition(in_dim=3, embed_dim=self.hidden_dim)
        elif vad_fusion_mode == "gate":
            gate_input_dim = 3 * 2 if use_text_vad else 3
            self.vad_gate = nn.Sequential(
                nn.Linear(gate_input_dim, 128),
                nn.GELU(),
                nn.Linear(128, 2 if use_text_vad else 1),
                nn.Softmax(dim=-1)
            )
        
        # 6. 分类器
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.LayerNorm(self.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim // 2, self.hidden_dim // 4),
            nn.LayerNorm(self.hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim // 4, num_labels)
        )
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        images: torch.Tensor = None,
        labels: torch.Tensor = None
    ):
        batch_size = input_ids.size(0)
        bart_outputs = self.bart.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        text_features = bart_outputs.last_hidden_state  # (batch, seq_len, hidden_dim)
        text_pooled = text_features[:, 0, :]  # (batch, hidden_dim)
        image_vad = None
        if images is not None:
            image_vad = self.image_vad_predictor(images)  # (batch, 3)
            image_vad_feat = self.image_vad_projection(image_vad)  # (batch, hidden_dim)
        else:
            image_vad = torch.zeros(batch_size, 3, device=input_ids.device)
            image_vad_feat = torch.zeros(batch_size, self.hidden_dim, device=input_ids.device)
        text_vad = None
        if self.use_text_vad:
            text_vad = self.text_vad_model(input_ids, attention_mask)  # (batch, 3)
            text_vad_feat = self.text_vad_projection(text_vad)  # (batch, hidden_dim)
        if self.vad_fusion_mode == "concat":
            if self.use_text_vad:
                fused_feat = torch.cat([text_pooled, image_vad_feat, text_vad_feat], dim=-1)
            else:
                fused_feat = torch.cat([text_pooled, image_vad_feat], dim=-1)
            fused_output = self.fusion_layer(fused_feat)
        elif self.vad_fusion_mode == "add":
            if self.use_text_vad:
                fused_output = text_pooled + image_vad_feat + text_vad_feat
            else:
                fused_output = text_pooled + image_vad_feat
        elif self.vad_fusion_mode == "film":
            gamma_img, beta_img = self.image_vad_film(image_vad)
            fused_output = gamma_img * text_pooled + beta_img
            if self.use_text_vad:
                gamma_txt, beta_txt = self.text_vad_film(text_vad)
                fused_output = gamma_txt * fused_output + beta_txt
        elif self.vad_fusion_mode == "gate":
            if self.use_text_vad:
                gate_input = torch.cat([image_vad, text_vad], dim=-1)
                gate_weights = self.vad_gate(gate_input)  # (batch, 2)
                fused_output = (text_pooled + 
                               gate_weights[:, 0:1] * image_vad_feat + 
                               gate_weights[:, 1:2] * text_vad_feat)
            else:
                gate_input = image_vad
                gate_weights = self.vad_gate(gate_input)  # (batch, 1)
                fused_output = text_pooled + gate_weights * image_vad_feat
        logits = self.classifier(fused_output)  # (batch, num_labels)
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
        return {
            'loss': loss,
            'logits': logits,
            'text_features': text_pooled,
            'image_vad': image_vad,
            'text_vad': text_vad,
            'fused_features': fused_output
        }

