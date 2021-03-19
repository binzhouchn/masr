import os
from config import pretrained_model_path
from models.conv import GatedConv
from record import record

model = GatedConv.load(os.path.join('..',pretrained_model_path))
record("../data_aishell/output.wav", time=5)  # modify time to how long you want

text = model.predict("../data_aishell/output.wav")

print("")
print("识别结果:")
print(text)
