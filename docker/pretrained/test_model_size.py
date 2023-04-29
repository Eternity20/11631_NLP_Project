import torch
import os
def print_size_of_model(model, label=""):

    torch.save(model, "temp.p")
    size=os.path.getsize("temp.p")
    print("model: ",label,' \t','Size (KB):', size/1e3)
    os.remove('temp.p')
    return size
