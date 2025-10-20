import matplotlib.pyplot as plt
import numpy as np

# pack или spread
filename = "mpi_results_pack.txt"
data = np.loadtxt(filename)

message_size = data[:,0]
bandwidth = data[:,2]

plt.figure(figsize=(8,5))
plt.plot(message_size, bandwidth, marker='o')
plt.xscale('log')
plt.xlabel("Msg Size (bytes)")
plt.ylabel("(MB/s)")
plt.title(f"MPI Bandwidth vs Message Size ({filename})")
plt.grid(True)
plt.show()
