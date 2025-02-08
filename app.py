import os
import json
import requests
import onnxruntime as ort
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
from io import BytesIO

app = FastAPI()

# 🌐 Đường dẫn model trên Google Drive (chỉnh sửa link này)
MODEL_URL = "https://drive.google.com/uc?export=download&id=1ABCD12345XYZ"
MODEL_PATH = "./image_caption_model.onnx"
VOCAB_PATH = "./tokenizer/vocab_fixed.json"  # Nếu bạn có file từ vựng

# 📥 Hàm tải model nếu chưa có
def download_model():
    if not os.path.exists(MODEL_PATH):
        print("📥 Downloading model from Google Drive...")
        response = requests.get(MODEL_URL, stream=True)
        with open(MODEL_PATH, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("✅ Model downloaded successfully!")

download_model()

# 🔍 Kiểm tra model
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Không tìm thấy mô hình tại: {MODEL_PATH}")

# 🔍 Kiểm tra file từ điển nếu có
if not os.path.exists(VOCAB_PATH):
    print(f"⚠️ Cảnh báo: Không tìm thấy từ điển tại {VOCAB_PATH}. API có thể hoạt động không chính xác.")
    vocab = {}
else:
    with open(VOCAB_PATH, "r", encoding="utf-8") as f:
        vocab = json.load(f)

# Đảm bảo key là int và value là str
id_to_word = {int(k): v for k, v in vocab.items()} if vocab else {}

# 🚀 Load mô hình ONNX
try:
    ort_session = ort.InferenceSession(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"❌ Lỗi khi load mô hình ONNX: {str(e)}")

def decode_tokens(token_ids):
    """Chuyển token ID thành câu caption."""
    if not isinstance(token_ids, list):
        token_ids = [token_ids]

    words = [id_to_word.get(token_id, "[UNK]") for token_id in token_ids]
    words = [word for word in words if word not in ["[PAD]", "[START]", "[END]", "[UNK]"]]
    caption = " ".join(words).capitalize() + "."
    return caption

def preprocess_image(image: Image.Image):
    """Resize và chuẩn hóa ảnh."""
    image = image.resize((224, 224))
    image_array = np.array(image).astype(np.float32) / 255.0
    image_array = np.transpose(image_array, (2, 0, 1))
    return np.expand_dims(image_array, axis=0)

@app.get("/")
def read_root():
    return {"message": "✅ API chạy thành công trên Render!"}

@app.post("/predict/")
async def predict(image: UploadFile = File(...)):
    try:
        contents = await image.read()
        img = Image.open(BytesIO(contents)).convert("RGB")
        processed_image = preprocess_image(img)

        dummy_caption = np.random.randint(0, len(id_to_word), (1, 10)).astype(np.int64)
        dummy_attention_mask = np.ones_like(dummy_caption).astype(np.int64)

        inputs = {
            "image": processed_image,
            "caption": dummy_caption,
            "attention_mask": dummy_attention_mask
        }

        outputs = ort_session.run(None, inputs)
        token_ids = np.argmax(outputs[0], axis=-1)[0]

        caption = decode_tokens(token_ids)
        return JSONResponse(content={"caption": caption})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
