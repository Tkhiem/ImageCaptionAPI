import os
import json
import gdown
import onnxruntime as ort
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
from io import BytesIO

app = FastAPI()

# 🔗 Đường dẫn model (Google Drive - Link tải trực tiếp)
MODEL_URL = "https://drive.google.com/uc?id=1W-WM4IaS_XXLzI6wgbiruzyuzoFkniSv"
MODEL_PATH = "./image_caption_model.onnx"
VOCAB_PATH = "./tokenizer/vocab.json"

# 📥 Hàm tải mô hình từ Google Drive nếu chưa có
def download_model():
    if not os.path.exists(MODEL_PATH):
        print("📥 Downloading model from Google Drive...")
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
        print("✅ Model downloaded successfully!")

download_model()

# 🛠 Kiểm tra sự tồn tại của mô hình
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Không tìm thấy mô hình tại: {MODEL_PATH}")

# 🔍 Kiểm tra file từ điển nếu có
if not os.path.exists(VOCAB_PATH):
    print(f"⚠️ Cảnh báo: Không tìm thấy từ điển tại {VOCAB_PATH}. API có thể hoạt động không chính xác.")
    vocab = {}
else:
    with open(VOCAB_PATH, "r", encoding="utf-8") as f:
        try:
            vocab = json.load(f)
            print(f"✅ Tải từ vựng thành công, tổng số từ: {len(vocab)}")
        except json.JSONDecodeError:
            raise ValueError(f"❌ Lỗi khi đọc file từ điển: {VOCAB_PATH}")

# 🗂 Đảm bảo vocab đúng định dạng {int: str}
try:
    id_to_word = {int(k): v for k, v in vocab.items()} if vocab else {}
except ValueError:
    raise ValueError("❌ Lỗi khi chuyển đổi vocab, có thể có key không phải số nguyên.")

# 🚀 Load mô hình ONNX
try:
    ort_session = ort.InferenceSession(MODEL_PATH)
    print("✅ Mô hình ONNX đã load thành công!")
except Exception as e:
    raise RuntimeError(f"❌ Lỗi khi load mô hình ONNX: {str(e)}")

# 🛠 Kiểm tra input/output của mô hình ONNX
print("📊 Tên đầu vào của mô hình:", [inp.name for inp in ort_session.get_inputs()])
print("📊 Tên đầu ra của mô hình:", [out.name for out in ort_session.get_outputs()])

# 📜 Hàm giải mã token thành caption
def decode_tokens(token_ids):
    if not isinstance(token_ids, list):
        token_ids = [token_ids]

    words = [id_to_word.get(token_id, "[UNK]") for token_id in token_ids]
    words = [word for word in words if word not in ["[PAD]", "[START]", "[END]", "[UNK]"]]
    caption = " ".join(words).capitalize() + "."
    return caption

# 📷 Hàm tiền xử lý ảnh
def preprocess_image(image: Image.Image):
    image = image.resize((224, 224))
    image_array = np.array(image).astype(np.float32) / 255.0
    image_array = np.transpose(image_array, (2, 0, 1))
    return np.expand_dims(image_array, axis=0)

# 📍 API kiểm tra kết nối
@app.get("/")
def read_root():
    return {"message": "✅ API chạy thành công trên Render!"}

# 🔥 API dự đoán caption từ ảnh
@app.post("/predict/")
async def predict(image: UploadFile = File(...)):
    try:
        contents = await image.read()
        img = Image.open(BytesIO(contents)).convert("RGB")
        print("📸 Ảnh nhận được!")

        processed_image = preprocess_image(img)
        print("✅ Ảnh đã được xử lý:", processed_image.shape)

        dummy_caption = np.random.randint(0, len(id_to_word), (1, 10)).astype(np.int64)
        dummy_attention_mask = np.ones_like(dummy_caption).astype(np.int64)

        print("📜 Dummy caption:", dummy_caption)
        print("📜 Attention mask:", dummy_attention_mask)

        # Kiểm tra tên đầu vào mô hình ONNX
        model_inputs = ort_session.get_inputs()
        input_names = [inp.name for inp in model_inputs]
        print("📊 Tên đầu vào mô hình:", input_names)

        # Tạo inputs đúng theo tên
        inputs = {
            input_names[0]: processed_image,
            input_names[1]: dummy_caption,
            input_names[2]: dummy_attention_mask
        }
        print("📤 Gửi dữ liệu vào mô hình:", {k: v.shape for k, v in inputs.items()})

        outputs = ort_session.run(None, inputs)
        print("📥 Nhận output từ mô hình:", [o.shape for o in outputs])

        # Kiểm tra output có đúng format không
        if len(outputs) == 0:
            raise ValueError("❌ Mô hình không trả về output nào!")

        token_ids = np.argmax(outputs[0], axis=-1)[0]
        print("📝 Token IDs:", token_ids)

        caption = decode_tokens(token_ids)
        print("🗣 Caption:", caption)

        return JSONResponse(content={"caption": caption})
    except Exception as e:
        print("❌ Lỗi:", str(e))
        return JSONResponse(status_code=500, content={"error": str(e)})

# 🌍 Chạy ứng dụng
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
