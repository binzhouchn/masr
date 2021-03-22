__mtime__= '20210318'
import torch
import feature
from models.conv import GatedConv
import torch.nn.functional as F
from ctcdecode import CTCBeamDecoder
from config import lm_path, pretrained_model_path

alpha = 0.8
beta = 0.3
cutoff_top_n = 40
cutoff_prob = 1.0
beam_width = 32
num_processes = 4
blank_index = 0

model = GatedConv.load(pretrained_model_path)
model.eval()

decoder = CTCBeamDecoder(
    model.vocabulary,
    lm_path,
    alpha,
    beta,
    cutoff_top_n,
    cutoff_prob,
    beam_width,
    num_processes,
    blank_index,
)


def translate(vocab, out, out_len):
    return "".join([vocab[x] for x in out[0:out_len]])


def predict(f):
    wav = feature.load_audio(f)
    spec = feature.spectrogram(wav)
    spec.unsqueeze_(0)
    with torch.no_grad():
        y = model.cnn(spec)
        y = F.softmax(y, 1)
    y_len = torch.tensor([y.size(-1)])
    y = y.permute(0, 2, 1)  # B * T * V
    print("decoding")
    out, score, offset, out_len = decoder.decode(y, y_len)
    return translate(model.vocabulary, out[0][0], out_len[0][0])

if __name__ == '__main__':
    #传入wav录音文件识别文本
    text = predict("data_aishell/BAC009S0765W0130.wav")
    print(text)