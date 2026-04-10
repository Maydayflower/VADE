
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
processor = BlipProcessor.from_pretrained("/workspace/models/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("/workspace/models/blip-image-captioning-base").to("cuda")


img_url = '/workspace/datasets/IJCAI2019_data/twitter2015_images/819411.jpg' 
raw_image = Image.open(img_url).convert('RGB')

# conditional image captioning
text = "a photography of"
inputs = processor(raw_image, text, return_tensors="pt").to("cuda")

out = model.generate(**inputs)
print(processor.decode(out[0], skip_special_tokens=True))
# >>> a photography of a woman and her dog

# unconditional image captioning
inputs = processor(raw_image, return_tensors="pt").to("cuda")

out = model.generate(**inputs)
print(processor.decode(out[0], skip_special_tokens=True))
# >>> a woman sitting on the beach with her dog
