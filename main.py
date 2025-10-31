import numpy as np
import faiss
import time
import psutil
import os
import csv
import gc
import matplotlib.pyplot as plt

# ===================== PARAMETERS =====================
D = 70  # vector dimension
QUERY_SIZE = 1000  # number of query vectors we test during run
DATASET_SIZES = [100_000, 300_000, 1_000_000]  # test dataset sizes to simulate them
NLIST_FACTORS = [1, 3, 5]  # controls IVF cluster scaling
M = 10  # number of PQ subvectors (each vector of dimension D is split into M chunks)
NBITS = 10  # bits per subquantizer as it defines the quantization precision
TOP_K = 10  # recall@10 (How many nearest neighbors to search for)
CSV_PATH = "benchmark_results.csv"
# ======================================================


# Helper: Measure memory usage
def get_memory_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


# Helper: Compute recall@k
def recall_at_k(true_I, test_I, k=10):
    n_queries = true_I.shape[0]
    correct = sum(
        len(set(true_I[i][:k]) & set(test_I[i][:k])) for i in range(n_queries)
    )
    return correct / (n_queries * k)


# Helper: Benchmark
def benchmark_index(index, xb, xq, label):
    gc.collect()
    mem_before = get_memory_mb()
    start = time.time()
    D, I = index.search(xq, TOP_K)
    elapsed = time.time() - start
    gc.collect()
    mem_after = get_memory_mb()
    return I, elapsed, max(mem_after - mem_before, 0)


results = []

for N in DATASET_SIZES:
    print(f"\n===== Testing with {N:,} vectors =====")

    # Generate dataset
    xb = np.random.random((N, D)).astype("float32")
    xq = np.random.random((QUERY_SIZE, D)).astype("float32")

    # Brute force baseline
    flat = faiss.IndexFlatL2(D)
    flat.add(xb)
    true_I, base_time, base_mem = benchmark_index(flat, xb, xq, "Flat")

    for c in NLIST_FACTORS:
        nlist = int(c * np.sqrt(N))
        nprobe = int(0.05 * nlist)  # 5% of nlist
        print(f"\n--- IVF+PQ: nlist={nlist}, nprobe={nprobe} ---")

        quantizer = faiss.IndexFlatL2(D)
        index = faiss.IndexIVFPQ(quantizer, D, nlist, M, NBITS)
        index.train(xb)
        index.add(xb)
        index.nprobe = nprobe

        test_I, ivfpq_time, ivfpq_mem = benchmark_index(index, xb, xq, "IVF+PQ")
        rec = recall_at_k(true_I, test_I, TOP_K)

        results.append(
            {
                "Dataset_Size": N,
                "nlist": nlist,
                "nprobe": nprobe,
                "Index_Type": "IVF+PQ",
                "Time_s": round(ivfpq_time, 3),
                "Recall@10": round(rec, 4),
                "Extra_Mem_MB": round(ivfpq_mem, 2),
            }
        )
        print(
            f"Recall@10 = {rec:.3f}, Time = {ivfpq_time:.3f}s, Extra Mem = {ivfpq_mem:.2f} MB"
        )

# Save results
with open(CSV_PATH, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

# Plot recall vs time
plt.figure(figsize=(8, 5))
for N in DATASET_SIZES:
    subset = [r for r in results if r["Dataset_Size"] == N]
    plt.plot(
        [r["Time_s"] for r in subset],
        [r["Recall@10"] for r in subset],
        marker="o",
        label=f"N={N/1000:.0f}K",
    )

plt.title("Recall vs Time for IVF+PQ")
plt.xlabel("Search Time (s)")
plt.ylabel("Recall@10")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("ivfpq_benchmark_plot.png")
plt.show()

print("\nResults saved to 'benchmark_results.csv' and plot generated.")
