from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from utils.utils import *

save_path = "main results/20250211_030807/"

train_ratio_range = np.round(np.arange(0.0, 1, 0.05), 2)
mu2_range = np.arange(1, 10)
results = []
for mu2 in mu2_range:
    results_path = f"{save_path}mu2({mu2})/"
    result = pd.read_csv(f"{results_path}results_mu2({mu2}).csv")
    results.append(result["accuracy"].to_list())

results = np.array(results).T
# pd.DataFrame(resutls, index = rho_min_range, columns=rho_plus_range).to_csv(f"{save_path}results_rho_min.csv")

# Create a DataFrame from the results
df = pd.DataFrame(results, columns=mu2_range, index=train_ratio_range)

# Save the DataFrame to a CSV file
df.to_csv(f"{save_path}results_mu2.csv")

fig_acc, ax_acc = plt.subplots()
for mu2, result in zip(mu2_range, results.T):
    alpha = 0.5 if mu2 != 5 else 1
    ax_acc.plot(
        train_ratio_range,
        result,
        "-*",
        # color="blue",
        label=f"Accuracy $\mu_2$={mu2}",
        alpha=alpha,
    )
ax_acc.set_ylabel("Accuracy")
ax_acc.set_xlabel("Train Ratio")
ax_acc.set_title("Accuracy")
ax_acc.legend(prop={"size": 8}, markerscale=0.0)
fig_acc.savefig(f"{save_path}accuracy2.png")
