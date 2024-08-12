import sys
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure


if len(sys.argv) > 1:
    address = sys.argv[1]
else:
    address = "/Users/aoroojlo/Downloads/slurm-30041.out"

text = open(address, "r", encoding = 'unicode_escape').read().split("\n")

losses = []
lrs = []
epochs = []

for line in text:
    if "epoch" in line and "loss" in line:
        try:
            tmp_text = line.split("'loss': ")[1]
            tmp_text = tmp_text.split(", 'learning_rate': ")
            loss = tmp_text[0]
            tmp_text = tmp_text[1].split(", 'epoch': ")
            lr = tmp_text[0]
            epoch = tmp_text[1].split("}")[0]
            losses += [float(loss)]
            lrs += [float(lr)]
            epochs += [float(epoch)]
        except:
            print(line)


figure(figsize=(8, 6), dpi=80)
plt.plot(epochs, losses, label="loss")
# plt.plot(epochs, lrs, label="lrs")
plt.legend(loc="best")
plt.grid(True)
plt.title("Loss vs Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig(address + "_loss.png")

