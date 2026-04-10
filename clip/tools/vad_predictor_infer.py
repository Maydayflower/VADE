from vad_predictor import VADPredictor
import torch
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('/workspace/models/bert-base-uncased')
model = VADPredictor()
model.load_state_dict(torch.load("/workspace/compare_model/bart/clip/vad_predictor.pt"))
model.eval()

texts = ["Two women work together, slicing and serving a freshly baked pizza under the open sky. ", 
"A half-eaten apple rests beside a vintage cell phone, evoking nostalgia and the passage of time.", 
"A serene, sunlit space where light dances through open doors and windows, inviting relaxation in its cozy, lived-in charm.",
"A zebra bends low, its nose touching the earth, seemingly searching for sustenance amidst a serene, sunlit setting.",
"An abandoned, decaying bathroom whispers tales of neglect, its sinks and soap dispensers hanging on by threads"]

# A store display filled with ripe, unripe bananas and other fruit. 
# A group of bananas surround a small display of kiwi.
# A fruit stand with plantains, kiwis, and bananas.
# A fruit stand that has bananas, papaya, and plantains.
# A fruit stand display with bananas and kiwi	
# A vibrant fruit stand brimming with ripe bananas, unripe plantains, and kiwis, inviting shoppers to indulge in nature's bounty.

#Two women work together, slicing and serving a freshly baked pizza under the open sky.
#A half-eaten apple rests beside a vintage cell phone, evoking nostalgia and the passage of time.
#A serene, sunlit space where light dances through open doors and windows, inviting relaxation in its cozy, lived-in charm.
#A zebra bends low, its nose touching the earth, seemingly searching for sustenance amidst a serene, sunlit setting.
#An abandoned, decaying bathroom whispers tales of neglect, its sinks and soap dispensers hanging on by threads


def predict_vad(model, tokenizer, texts):
    model.eval()
    with torch.no_grad():
        encoded = tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=10,
            return_tensors='pt'
        )
        input_ids = encoded['input_ids']
        attention_mask = encoded['attention_mask']
        outputs = model(input_ids, attention_mask)
        return outputs.cpu().numpy()

vad_preds = predict_vad(model, tokenizer, texts)
for w, v in zip(texts, vad_preds):
    print(f"{w}: valence={v[0]:.3f}, arousal={v[1]:.3f}, dominance={v[2]:.3f}")