import pandas as pd
import matplotlib.pyplot as plt
import os
import json


for file in ["hlg.json", "reg.json"]:
    path = os.path.join("data", "Alien", file)
    with open(path, "r") as in_file:
        data = json.load(in_file)
    df = pd.DataFrame(data)
    plt.plot(df["val_mae"], label=file[:3])
plt.legend()
plt.ylabel("MAE")
plt.xlabel("Train Batches (x $10^4$)")
plt.title("Validation MAE over Time")
plt.show()