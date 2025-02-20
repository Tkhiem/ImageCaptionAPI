import torch
import torch.nn as nn
from torchvision import models

class ImageCaptionModel(nn.Module):
    def __init__(self, model_name="swin_v2_s", embedding_dim=512, vocab_size=10000, 
                 num_heads_decoder=8, num_transformer_decoder_layers=6, ffn_dim=2048, dropout_decoder=0.1):
        super(ImageCaptionModel, self).__init__()

        # Swin Transformer Encoder
        self.encoder = models.swin_v2_s(weights="IMAGENET1K_V1")

        # Lấy số chiều đầu ra của encoder trước khi thay thế head
        encoder_dim = self.encoder.head.in_features  # Fix lỗi

        # Thay thế head bằng Identity
        self.encoder.head = nn.Identity()
        self.encoder_avgpool = nn.AdaptiveAvgPool2d((1, 1))  # Đảm bảo đầu ra có dạng [B, C, 1, 1]

        self.feature_proj = nn.Linear(encoder_dim, embedding_dim)  # Chuyển sang embedding_dim
        
        # Embedding Layer cho Caption
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(d_model=embedding_dim, nhead=num_heads_decoder, 
                                                   dim_feedforward=ffn_dim, dropout=dropout_decoder, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_transformer_decoder_layers)
        
        # Projection lên không gian từ vựng
        self.fc = nn.Linear(embedding_dim, vocab_size)
    
    def forward(self, images, captions):
        # Feature extraction từ ảnh
        enc_out = self.encoder(images)  # [B, C, H, W]
        enc_out = self.encoder_avgpool(enc_out).squeeze(-1).squeeze(-1)  # [B, C]
        enc_out = self.feature_proj(enc_out).unsqueeze(1)  # [B, 1, embed_dim]

        # Chuyển đổi caption thành vector nhúng
        captions_emb = self.embedding(captions)  # [B, seq_len, embed_dim]
        
        # Transformer Decoder
        dec_out = self.decoder(captions_emb, enc_out)  # [B, seq_len, embed_dim]
        
        # Dự đoán từ vựng
        return self.fc(dec_out)  # [B, seq_len, vocab_size]

    @staticmethod
    def load_model(model_path, device="cpu", **kwargs):
        model = ImageCaptionModel(**kwargs)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        return model
