import os
import json
import torch
import requests
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from io import BytesIO
from model import ImageCaptionModel  # Import model từ model.py
from transformers import AutoTokenizer  # Import tokenizer

app = FastAPI()

# 📥 Định nghĩa đường dẫn
MODEL_PATH = "./swinv2_transformerdecoder.pth"
TOKENIZER_PATH = "./tokenizer"
HUGGINGFACE_MODEL_URL = "https://huggingface.co/NguyenKhiem/SwinV2Transformer/resolve/main/swinv2_transformerdecoder.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 📥 Tải mô hình từ Hugging Face nếu chưa có
if not os.path.exists(MODEL_PATH):
    print("🔄 Đang tải mô hình từ Hugging Face...")
    response = requests.get(HUGGINGFACE_MODEL_URL)
    if response.status_code == 200:
        with open(MODEL_PATH, "wb") as f:
            f.write(response.content)
        print("✅ Tải mô hình thành công!")
    else:
        raise FileNotFoundError(f"❌ Không thể tải mô hình, mã lỗi: {response.status_code}")

# Khởi tạo model
model = ImageCaptionModel()  
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))  # Load trọng số
model.to(DEVICE)
model.eval()  # Chuyển sang chế độ đánh giá

# 📜 Tải tokenizer
if not os.path.exists(TOKENIZER_PATH):
    raise FileNotFoundError(f"❌ Không tìm thấy tokenizer tại: {TOKENIZER_PATH}")

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)  # Load tokenizer

# 📜 Hàm giải mã token thành caption
def decode_tokens(token_ids):
    words = tokenizer.decode(token_ids, skip_special_tokens=True)  # Dùng tokenizer để giải mã
    return words.capitalize() + "."

# 📷 Tiền xử lý ảnh
def preprocess_image(image: Image.Image):
    image = image.resize((224, 224))
    image_array = np.array(image).astype(np.float32) / 255.0
    image_array = np.transpose(image_array, (2, 0, 1))
    image_tensor = torch.tensor(image_array, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    return image_tensor

# 📍 API kiểm tra kết nối
@app.get("/")
def read_root():
    return {"message": "✅ API đang chạy thành công trên Render!"}

# 🔥 API dự đoán caption từ ảnh
@app.post("/predict/")
async def predict(image: UploadFile = File(...)):
    try:
        contents = await image.read()
        img = Image.open(BytesIO(contents)).convert("RGB")

        # Tiền xử lý ảnh
        image_tensor = preprocess_image(img)

        # Tạo đầu vào ban đầu cho Transformer Decoder (thường là token <START>)
        start_token = torch.tensor([tokenizer.bos_token_id], dtype=torch.long).unsqueeze(0).to(DEVICE)

        # Chạy mô hình để tạo caption
        with torch.no_grad():
            generated_tokens = model(image_tensor, start_token)

        # Lấy token có xác suất cao nhất
        token_ids = torch.argmax(generated_tokens, dim=-1).cpu().numpy()[0]

        # Giải mã token thành caption
        caption = decode_tokens(token_ids)

        return JSONResponse(content={"caption": caption})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
