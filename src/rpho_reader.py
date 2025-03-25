from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

save_path = "results/20250210_233416/"

results = []
rho_plus_range = np.arange(1, 10.5, 0.5)
rho_min_range = np.round(np.arange(0.1, 1.1, 0.1), 1)
for rho_min in rho_min_range:
    results_path = f"{save_path}rhomin({rho_min:0.1f})/"
    result = pd.read_csv(f"{results_path}results_rhoplus.csv")
    results.append(result["accuracy"].to_list())

# pd.DataFrame(resutls, index = rho_min_range, columns=rho_plus_range).to_csv(f"{save_path}results_rho_min.csv")

# Create a DataFrame from the results
df = pd.DataFrame(results, index=rho_min_range, columns=rho_plus_range)

# Save the DataFrame to a CSV file
df.to_csv(f"{save_path}results_rho_min.csv")

# Calculate sigma_signal/sigma_noise
sigma_noise = df.index.values
sigma_signal_over_sigma_noise = df.columns.values

# Plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(
    df,
    annot=True,
    fmt=".2f",
    cmap="viridis",
    xticklabels=rho_plus_range,
    yticklabels=rho_min_range,
)
plt.xlabel("$\\frac{\\rho^+}{\\rho^-}$")
plt.ylabel("$\\rho^-$")
plt.title("Heatmap of Accuracy")
plt.show()
