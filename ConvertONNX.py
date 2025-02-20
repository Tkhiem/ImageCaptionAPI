import torch
import onnx
from model import ImageCaptionModel

# Định nghĩa đường dẫn
MODEL_PATH = "swinv2_transformerdecoder.pth"
ONNX_MODEL_PATH = "swinv2_transformerdecoder.onnx"

# Kiểm tra thiết bị
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Khởi tạo mô hình
model = ImageCaptionModel()
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# Dummy input
dummy_image = torch.randn(1, 3, 224, 224).to(DEVICE)  # Ảnh đầu vào
dummy_caption = torch.randint(0, 10000, (1, 10)).to(DEVICE)  # Câu đầu vào giả lập

# Trích xuất đặc trưng ảnh từ encoder
with torch.no_grad():
    image_features = model.encoder(dummy_image)
    if image_features.dim() == 4:  # Swin Transformer output thường là [B, C, H, W]
        image_features = model.encoder_avgpool(image_features).flatten(1)  # Chuyển về [B, C]

# Xuất sang ONNX
torch.onnx.export(
    model,
    (image_features, dummy_caption),  # Tuple đầu vào
    ONNX_MODEL_PATH,
    input_names=["image_features", "captions"], 
    output_names=["output"],
    dynamic_axes={
        "image_features": {0: "batch_size"},
        "captions": {0: "batch_size", 1: "sequence_length"},
        "output": {0: "batch_size", 1: "sequence_length"}
    },
    opset_version=12
)

print(f"✅ Mô hình đã được chuyển sang ONNX và lưu tại {ONNX_MODEL_PATH}")
