import requests
import base64

def encode_img(image_path):
    with open(image_path, 'rb') as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

url = "https://api.siliconflow.cn/v1/chat/completions"
captions_for_prompt = "Through a fish eye lens, a man about to bite into a sandwich | A MAN SITTING AT A TABLE EATING A SANDWICH | The smiling man is taking a bite of a sandwich. | a man sitting at a table so he can eat a sandwich | A peep hole view of a a man biting a sandwich."
prompt_text = (
            f"Here are several different descriptions of the image: {captions_for_prompt} "
            f"Please use them as reference to generate a single, more emotional caption for the image. "
            f"Add suitable adjectives or verbs, make it concise and less than 20 words."
        )
payload = {
    "model": "Qwen/Qwen3-VL-30B-A3B-Instruct",
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt_text
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{encode_img('/workspace/compare_model/bart/clip/data/saved_image.jpg')}"
                    }
                }
            ]
        }
    ]
}
headers = {
    "Authorization": "YOUR-OWN-API",
    "Content-Type": "application/json"
}

response = requests.post(url, json=payload, headers=headers)

print(response.json()['choices'][0]['message']['content'].strip())