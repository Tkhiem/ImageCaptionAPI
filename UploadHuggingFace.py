from huggingface_hub import HfApi

# Thay YOUR_USERNAME và YOUR_MODEL_NAME bằng thông tin của bạn
REPO_ID = "NguyenKhiem/SwinV2Transformer"
MODEL_PATH = "./swinv2_transformerdecoder.pth"  # Đường dẫn file model .pth

# Khởi tạo API
api = HfApi()

# Đẩy file lên Hugging Face Hub
api.upload_file(
    path_or_fileobj=MODEL_PATH,
    path_in_repo="swinv2_transformerdecoder.pth",  # Tên file khi upload lên
    repo_id=REPO_ID,
    repo_type="model"
)

print("✅ Model đã được upload thành công lên Hugging Face Hub!")
