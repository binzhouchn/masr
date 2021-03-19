__mtime__= '20210318'
import os
from models.conv import GatedConv
from config import pretrained_model_path

model = GatedConv.load(os.path.join('..', pretrained_model_path))

text = model.predict("../data_aishell/BAC009S0765W0130.wav")

print("")
print("识别结果:")
print(text)
