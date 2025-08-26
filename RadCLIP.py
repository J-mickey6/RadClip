import os
import random
import transformers
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor, CLIPModel
from torch import nn, optim
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch

# 2. 텍스트 프롬프트 설정
text_prompts = {
    "PD": [
        "MRI of a brain with Parkinson's disease",
        "Axial T1-weighted brain scan of a Parkinson’s patient",
        "Neuroimaging showing Parkinson's pathology",
        "Abnormal MRI scan suggesting Parkinson’s",
        "MRI of a brain affected by neurodegeneration"
    ],
    "PDX": [
        "Normal brain MRI",
        "MRI of a healthy human brain",
        "Typical neuroimaging scan with no abnormalities",
        "Control subject brain MRI",
        "MRI showing no pathological signs"
    ]
}

# 1. Dataset 클래스 정의
# CLIP 모델은 contrastive learning 방식으로 end-to-end 학습에서는 image+text쌍으로 묶어서 processor에 들어가야함.
class MRIRadCLIPDataset(Dataset):
    def __init__(self, samples, processor, text_prompts):
        self.samples = samples  # (image_path, label) list
        self.processor = processor
        self.text_prompts = text_prompts

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        image = Image.open(image_path).convert("RGB")

        text = random.choice(self.text_prompts[label])
        inputs = self.processor(text=text, images=image, return_tensors="pt", padding=True)
        return {k: v.squeeze(0) for k, v in inputs.items()}

# 2. 전체 sample 수집
root_dir = "C:/visual code/MRI/dataset/MRI_train"
all_samples = []
for label in text_prompts.keys():
    dir_path = os.path.join(root_dir, label)
    for fname in os.listdir(dir_path):
        if fname.endswith((".jpg", ".png")):
            image_path = os.path.join(dir_path, fname)
            all_samples.append((image_path, label))


# 5. 모델 및 processor 불러오기
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to("cuda")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

# 내부 모듈 접근
text_encoder = model.text_model
vision_encoder = model.vision_model
text_projector = model.text_projection
vision_projector = model.visual_projection

# freeze 설정(TextEncoder, TextProjector, VisionEncoder)
for param in text_encoder.parameters():
    param.requires_grad = False
for param in text_projector.parameters():
    param.requires_grad = False
for param in vision_encoder.parameters():
    param.requires_grad = False
# vision_projector🔥는 학습 대상이므로 freeze 안 함


# 6. Dataset & DataLoader 구성
train_dataset = MRIRadCLIPDataset(train_samples, processor, text_prompts)
test_dataset = MRIRadCLIPDataset(test_samples, processor, text_prompts)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)





# 7. 학습 설정
optimizer = optim.AdamW(model.parameters(), lr=5e-6)
loss_fn = nn.CrossEntropyLoss()

# 8. 학습 루프
for epoch in range(5):
    model.train()
    total_loss = 0

    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        input_ids = batch["input_ids"].to("cuda")
        attention_mask = batch["attention_mask"].to("cuda")
        pixel_values = batch["pixel_values"].to("cuda")

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values)
        logits_per_image = outputs.logits_per_image
        logits_per_text = outputs.logits_per_text

        labels = torch.arange(len(logits_per_image)).to("cuda")
        loss = (loss_fn(logits_per_image, labels) + loss_fn(logits_per_text, labels)) / 2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1} Loss: {total_loss / len(train_loader):.4f}")

# 9. 모델 저장
model.save_pretrained("C:/visual code/MRI/radclip_model")
processor.save_pretrained("C:/visual code/MRI/radclip_model")

# 10. 평가 함수
def classify_text_to_label(text, text_prompts):
    for label, prompts in text_prompts.items():
        if text in prompts:
            return label
    return None

def evaluate_model(model, processor, image_paths, ground_truths, text_prompts):
    model.eval()
    correct = 0
    total = 0

    texts = [prompt for label in text_prompts for prompt in text_prompts[label]]
    text_inputs = processor(text=texts, return_tensors="pt", padding=True).to("cuda")

    with torch.no_grad():
        text_features = model.get_text_features(**text_inputs)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        for img_path, true_label in zip(image_paths, ground_truths):
            image = Image.open(img_path).convert("RGB")
            image_inputs = processor(images=image, return_tensors="pt").to("cuda")
            image_features = model.get_image_features(**image_inputs)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            similarity = torch.matmul(image_features, text_features.T)
            predicted_index = similarity.argmax().item()
            predicted_label = classify_text_to_label(texts[predicted_index], text_prompts)

            if predicted_label == true_label:
                correct += 1
            total += 1

    accuracy = correct / total
    return accuracy

# 11. 테스트 평가 실행
test_accuracy = evaluate_model(
    model, processor, test_paths, test_labels, text_prompts
)

print(f"\n✅ Test Accuracy: {test_accuracy*100:.2f}%")
