import os
import pandas as pd
import json
import matplotlib.pyplot as plt


games = ["Breakout", "Pong", "Qbert", "SpaceInvaders", "Centipede", "Boxing", "MsPacman", "PrivateEye", "KungFuMaster", "Jamesbond", "Tutankham"]
base_dir = os.path.join("data", "test_rerun")
# keys = ["reg_mae", "reg_mse", "reg_loss", "hl_mae", "hl_mse", "hl_loss", "game"]
# results = {key: [] for key in keys}
# for game in games:
#     file_path = os.path.join(base_dir, game, "results.json")
#     with open(file_path, "r") as in_file:
#         data = json.load(in_file)
#     for key, val in data.items():
#         results[key].append(val)
#     results["game"].append(game)

# df = pd.DataFrame(results)
# print(df)
fig, axs = plt.subplots(3, 4, figsize=(19, 9), layout="constrained")
keys = ["hl_mae", "hl_mse", "hl_loss", "reg_mae", "reg_mse", "reg_loss", "game"]
results = {key:[] for key in keys}
i = 0
for game in games:
    reg_path = os.path.join(base_dir, game, "reg.json")
    with open(reg_path, "r") as in_file:
        reg_data = json.load(in_file)
    reg_df = pd.DataFrame(reg_data)

    hl_path = os.path.join(base_dir, game, "hlg.json")
    with open(hl_path, "r") as in_file:
        hl_data = json.load(in_file)
    hl_df = pd.DataFrame(hl_data)
    
    reg_mean = reg_df.mean()
    hl_mean = hl_df.mean()

    for key in ["mae", "mse", "loss"]:
        results[f"reg_{key}"].append(reg_mean[key])
        results[f"hl_{key}"].append(hl_mean[key])
    results["game"].append(game)

    ax = axs[i // 4, i % 4]
    ax.plot(hl_df["val_mae"], label="HL-Gaussian", color="tab:orange")
    ax.plot(reg_df["val_mae"], label="$\ell_2$", color="tab:green")
    ax.title.set_text(game)
    i += 1

handles, labels = axs[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower right')
plt.show()
res_df = pd.DataFrame(results)
print(res_df)