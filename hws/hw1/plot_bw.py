import matplotlib.pyplot as plt
import numpy as np

files = ["mpi_results_pack.txt", "mpi_results_spread.txt"]

plt.figure(figsize=(8,5))

for f in files:
    data = np.loadtxt(f)
    message_size = data[:,0]
    bandwidth = data[:,2]
    plt.plot(message_size, bandwidth, marker='o', label=f)

plt.xscale('log')
plt.xlabel("Message size (bytes)")
plt.ylabel("Bandwidth (MB/s)")
plt.title("MPI P2P Communication Benchmark")
plt.grid(True)
plt.legend()
plt.savefig("bandwidth_comparison.png")
plt.show()
