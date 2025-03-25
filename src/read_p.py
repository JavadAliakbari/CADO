from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

save_path = "main results/20250211_033908/"

q_vals = np.round(np.arange(0.0, 1, 0.02), 2)
num_atoms_range = np.arange(1, 7)
results0 = []
results1 = []
results2 = []
for num_atoms in range(1, 7):
    results_path = f"{save_path}num_atoms({num_atoms})/"
    result0 = pd.read_csv(f"{results_path}results_u0(0.0)_u1(0).csv")
    result1 = pd.read_csv(f"{results_path}results_u0(0.0)_u1(1).csv")
    result2 = pd.read_csv(f"{results_path}results_u0(0.05)_u1(1).csv")
    results0.append(result0["accuracy"].to_list())
    results1.append(result1["accuracy"].to_list())
    results2.append(result2["accuracy"].to_list())


# pd.DataFrame(resutls, index = rho_min_range, columns=rho_plus_range).to_csv(f"{save_path}results_rho_min.csv")

# Create a DataFrame from the results
df0 = pd.DataFrame(results0, index=num_atoms_range, columns=q_vals).T
df1 = pd.DataFrame(results1, index=num_atoms_range, columns=q_vals).T
df2 = pd.DataFrame(results2, index=num_atoms_range, columns=q_vals).T

# Save the DataFrame to a CSV file
df0.to_csv(f"{save_path}results_u0(0.0)_u1(0).csv")
df1.to_csv(f"{save_path}results_u0(0.0)_u1(1).csv")
df2.to_csv(f"{save_path}results_u0(0.05)_u1(1).csv")

# # Calculate sigma_signal/sigma_noise
# sigma_noise = df.index.values
# sigma_signal_over_sigma_noise = df.columns.values

# # Plot the heatmap
# plt.figure(figsize=(10, 8))
# sns.heatmap(
#     df,
#     annot=True,
#     fmt=".2f",
#     cmap="viridis",
#     xticklabels=rho_plus_range,
#     yticklabels=rho_min_range,
# )
# plt.xlabel("$\\frac{\\rho^+}{\\rho^-}$")
# plt.ylabel("$\\rho^-$")
# plt.title("Heatmap of Accuracy")
# plt.show()
