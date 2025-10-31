import nbformat as nbf
from zipfile import ZipFile
import os

# Create notebook object
nb = nbf.v4.new_notebook()

# Mega stress test cells
cells = [
    nbf.v4.new_markdown_cell("""# Binder Mega Stress Test\nThis notebook runs multiple heavy computations to fully stress Binder -> Docker -> Kubernetes environment, while showing CPU and memory usage."""),
    
    nbf.v4.new_code_cell("""\
import numpy as np
from numba import njit, prange
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import time, multiprocessing, psutil

print('CPUs available:', multiprocessing.cpu_count())
"""),

    nbf.v4.new_code_cell("""\
# Helper function to print CPU/memory
def show_sys_stats():
    print('CPU % per core:', psutil.cpu_percent(interval=1, percpu=True))
    print('Memory % used:', psutil.virtual_memory().percent)
"""),

    nbf.v4.new_code_cell("""\
# 1. Sequential large matrix multiplications
@njit(parallel=True)
def big_dot(A, B):
    n = A.shape[0]
    C = np.zeros((n, n))
    for i in prange(n):
        for j in range(n):
            for k in range(n):
                C[i, j] += A[i, k] * B[k, j]
    return C

sizes = [1000, 1200]
for n in sizes:
    A = np.random.rand(n, n)
    B = np.random.rand(n, n)
    start = time.time()
    C = big_dot(A, B)
    elapsed = time.time() - start
    print(f"Matrix {n}x{n} multiplication elapsed: {elapsed:.2f}s")
    show_sys_stats()
    plt.figure(figsize=(6,5))
    plt.imshow(C[:100,:100], cmap='viridis')
    plt.title(f"{n}x{n} Matrix Product Snapshot")
    plt.suptitle(f"Elapsed: {elapsed:.2f}s", y=0.95)
    plt.colorbar()
    plt.show()
"""),

    nbf.v4.new_code_cell("""\
# 2. Parallel matrix multiplications
def mat_task(seed):
    np.random.seed(seed)
    A = np.random.rand(800, 800)
    B = np.random.rand(800, 800)
    return big_dot(A, B)

start = time.time()
results = Parallel(n_jobs=-1)(delayed(mat_task)(i) for i in range(4))
elapsed = time.time() - start
print(f"Parallel 4x 800x800 matrices elapsed: {elapsed:.2f}s")
show_sys_stats()

plt.figure(figsize=(10,2))
for i, R in enumerate(results):
    plt.subplot(1,4,i+1)
    plt.imshow(R[:50,:50], cmap='plasma')
    plt.title(f"Task {i+1}")
plt.suptitle(f"Snapshots of parallel matrix products - elapsed: {elapsed:.2f}s", y=1.05)
plt.show()
"""),

    nbf.v4.new_code_cell("""\
# 3. Monte Carlo π estimation
@njit(parallel=True)
def monte_carlo_pi(num_samples):
    count = 0
    for i in prange(num_samples):
        x, y = np.random.rand(), np.random.rand()
        if x**2 + y**2 <= 1:
            count += 1
    return count

samples = 300_000_000
start = time.time()
hits = monte_carlo_pi(samples)
pi_est = 4 * hits / samples
elapsed = time.time() - start
print(f"Monte Carlo π estimation: {pi_est:.6f}, elapsed: {elapsed:.2f}s")
show_sys_stats()
"""),

    nbf.v4.new_code_cell("""\
# 4. SVD computation
from scipy.linalg import svd
X = np.random.rand(1000, 1000)
start = time.time()
U, s, V = svd(X)
elapsed = time.time() - start
print(f"SVD of 1000x1000 matrix elapsed: {elapsed:.2f}s")
show_sys_stats()
plt.figure(figsize=(6,4))
plt.plot(s[:50], marker='o')
plt.title("Top 50 singular values of 1000x1000 matrix")
plt.suptitle(f"Elapsed: {elapsed:.2f}s", y=0.95)
plt.xlabel("Index")
plt.ylabel("Singular value")
plt.grid(True)
plt.show()
"")
]

# Add cells to notebook
nb['cells'] = cells

# Save notebook
repo_dir = "./binder_mega_stress"
os.makedirs(repo_dir, exist_ok=True)
notebook_path = os.path.join(repo_dir, "mega_stress_test.ipynb")
with open(notebook_path, 'w') as f:
    nbf.write(nb, f)

# Add environment.yml
env_yml = """name: binder-stress-mega
channels:
  - conda-forge
dependencies:
  - python=3.11
  - numpy
  - scipy
  - matplotlib
  - numba
  - pandas
  - scikit-learn
  - joblib
  - ipywidgets
  - jupyterlab
  - psutil
  - pip
  - pip:
      - tqdm
"""
with open(os.path.join(repo_dir, "environment.yml"), 'w') as f:
    f.write(env_yml)

# Optional postBuild
postbuild = """#!/bin/bash\
echo "Running postBuild setup for mega_stress_test..." \
python -m numba -s | head -n 20 \
echo "postBuild complete." \
"""
with open(os.path.join(repo_dir, "postBuild"), 'w') as f:
    f.write(postbuild)

# Create ZIP
zip_path = "./binder_mega_stress.zip"
with ZipFile(zip_path, 'w') as zipf:
    for foldername, _, filenames in os.walk(repo_dir):
        for filename in filenames:
            filepath = os.path.join(foldername, filename)
            arcname = os.path.relpath(filepath, repo_dir)
            zipf.write(filepath, arcname)

# zip_path
