import torch
import torch.nn as nn
from torchvision import models

class ImageCaptionModel(nn.Module):
    def __init__(self, model_name="swin_v2_s", embedding_dim=512, vocab_size=10000, 
                 num_heads_decoder=8, num_transformer_decoder_layers=6, ffn_dim=2048, dropout_decoder=0.1):
        super(ImageCaptionModel, self).__init__()

        # Swin Transformer Encoder
        self.encoder = models.swin_v2_s(weights="IMAGENET1K_V1")
        self.encoder.head = nn.Identity()  # Loại bỏ lớp classification

        # Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(d_model=embedding_dim, nhead=num_heads_decoder, 
                                                   dim_feedforward=ffn_dim, dropout=dropout_decoder, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_transformer_decoder_layers)

        # Linear projection to vocabulary size
        self.fc = nn.Linear(embedding_dim, vocab_size)

    def forward(self, image_features, captions):
        enc_out = self.encoder(image_features)
        dec_out = self.decoder(captions, enc_out)
        return self.fc(dec_out)

    @staticmethod
    def load_model(model_path, device="cpu"):
        model = ImageCaptionModel()
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        return model
