from transformers import CLIPModel, CLIPProcessor
import torch.nn as nn
from transformers import BertModel, BertTokenizer

model = CLIPModel.from_pretrained("/workspace/models/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("/workspace/models/clip-vit-base-patch32")

def get_clip_features(image):
    inputs = processor(text=["a photo of a cat"], images=image, return_tensors="pt")
    outputs = model(**inputs)
    return outputs.pooler_output

def get_clip_text_features(text):
    inputs = processor(text=text, return_tensors="pt")
    outputs = model(**inputs)
    return outputs.pooler_output

class VADModel(nn.Module):
    def __init__(self, bert_model_name='/workspace/models/bert-base-uncased', output_dim=3):
        super(VADModel, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
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
            nn.Linear(64, output_dim)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # 取[CLS]向量
        cls_output = outputs.last_hidden_state[:, 0, :]
        out = self.regressor(cls_output)
        return out


vad_model = VADModel()