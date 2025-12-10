import pandas as pd
import matplotlib.pyplot as plt

# Load CSV
df = pd.read_csv("/Users/lydialichen/Library/CloudStorage/OneDrive-UniversityCollegeLondon/Year 3/Research Project in Biomedical Engineering/Code/DTOFs_whiteMC(in).csv")

# Time vector (ns)
time_ns = df["time_ns"].values

# DTOFs as matrix [n_time_bins x n_curves]
tpsf_matrix = df.drop(columns=["time_ns"]).values

# Columns names with μa / μs' info
headers = df.columns[1:]

num_mua = 20 

plt.figure()
for col in range(0, tpsf_matrix.shape[1], num_mua):
    plt.semilogy(time_ns, tpsf_matrix[:, col], label=f"DTOF {col}")

plt.xlabel("Time (ns)")
plt.ylabel("Intensity (a.u.)")
plt.title("One DTOF per μs'")
plt.grid(True)
plt.legend()
plt.show()
