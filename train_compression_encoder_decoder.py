import clip
import torch
import pandas as pd
import torch.nn as nn
import time
import torch.nn.functional as F

from lseg import LSegNet
import os

csv_file = './objectInfo150.csv'
if os.path.exists(csv_file):
    df = pd.read_csv(csv_file)
else:
    raise FileNotFoundError("objectInfo150.csv does not exist. Please download from https://github.com/CSAILVision/sceneparsing/blob/master/objectInfo150.csv")

labels = df['Name'].values
new_labels = []
for label in labels:
    new_labels.extend(list(label.split(";")))
labels = new_labels

device = "cuda" if torch.cuda.is_available() else "cpu"

net = LSegNet(
        backbone="clip_vitl16_384",
        features=256,
        crop_size=480,
        arch_option=0,
        block_depth=0,
        activation="lrelu",
    )
# Load pre-trained weights
net.load_state_dict(torch.load("./checkpoints/lseg_minimal_e200.ckpt", map_location=device))
net.eval()
net.to(device)

clip_text_encoder = net.clip_pretrained.encode_text

prompts = labels

prompt = clip.tokenize(prompts)
prompt = prompt.cuda()

text_feat = clip_text_encoder(prompt)  # N, 512, N - number of prompts
text_feat_norm = torch.nn.functional.normalize(text_feat, dim=1)
text_feat_norm = text_feat_norm.float().to(device)
print(text_feat_norm.shape)


class EncoderDecoder(nn.Module):
    def __init__(self):
        super(EncoderDecoder, self).__init__()

        # Same thing can be realized by linear layer or 1x1 conv layer
        # Defines the layer as a matrix to avoid reshaping
        # input dim shape (any_number_of_dimensions..., 512)
        self.encoder = nn.Parameter(torch.randn(512, 16))
        self.decoder = nn.Parameter(torch.randn(16, 512))


    def forward(self, x):
        x = x @ self.encoder
        y = x @ self.decoder
        return x, y


model = EncoderDecoder().to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-4)
text_feat_norm = text_feat_norm.detach()




def latent_cosine_preservation_loss(z, x):
    z_norm = F.normalize(z, dim=1)
    x_norm = F.normalize(x, dim=1)

    cosine_z = z_norm @ z_norm.T
    cosine_x = x_norm @ x_norm.T

    return F.mse_loss(cosine_z, cosine_x)


t1 = time.time()
for i in range(100000):
    x, y = model(text_feat_norm)
    y = torch.nn.functional.normalize(y, dim=1)
    loss1 = torch.nn.functional.mse_loss(text_feat_norm, y)
    loss2 = latent_cosine_preservation_loss(x, text_feat_norm)
    loss = loss1 + loss2
    loss.backward()
    opt.step()
    opt.zero_grad()
    if i % 1000 == 0:
        print(loss.item())


t2 = time.time()
print("Time taken for training encoder decoder model: ", t2 - t1)

torch.save(model.state_dict(), "encoder_decoder.ckpt")


# How to use
# model = EncoderDecoder()
# compressed = uncompressed @ model.encoder
# uncompressed = compressed @ model.decoder
# uncompressed tensor should have the last dimension as 512