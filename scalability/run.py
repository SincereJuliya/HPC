import os
import re
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def ds_key(name):
    """Sorting of the names per plots"""
    num_part = re.sub(r'[^\d.]', '', name)
    if not num_part: return 0
    num = float(num_part)
    if 'M' in name: num *= 1000
    return num

def parse_all_logs(base_dir):
    data = []
    # file folders
    folders = ['mpi_strong', 'hybrid_strong', 'mpi_weak', 'hybrid_weak']
    
    for folder in folders:
        path = os.path.join(base_dir, folder)
        if not os.path.exists(path):
            print(f"Warning: Path {path} not found.")
            continue
            
        mode = 'Hybrid' if 'hybrid' in folder else 'MPI'
        test_type = 'Strong' if 'strong' in folder else 'Weak'
        
        for filename in os.listdir(path):
            if filename.endswith(".txt"):
                # searching for the dataset size
                ds_match = re.search(r'dataset_([^_.]+)', filename)
                dataset_name = ds_match.group(1) if ds_match else "unknown"
                
                with open(os.path.join(path, filename), 'r') as f:
                    content = f.read()
                    time_match = re.search(r'Total execution time:\s+([\d.]+)\s+s', content)
                    # search for the ranks
                    ranks_match = re.search(r'Ranks:\s+(\d+)', content)
                    
                    if time_match and ranks_match:
                        data.append({
                            'Mode': mode, 
                            'Type': test_type,
                            'Processes': int(ranks_match.group(1)),
                            'Time': float(time_match.group(1)),
                            'Dataset': dataset_name
                        })
    return pd.DataFrame(data)

def generate_plots_and_tables(df):
    if df.empty:
        print("No data to process!")
        return

    # --- 1. STRONG SCALING (Dataset 2M) ---
    print("\n" + "="*50)
    print("TABLE: STRONG SCALING EXECUTION TIME (N=2M)")
    print("="*50)
    strong_2m = df[(df['Type'] == 'Strong') & (df['Dataset'] == '2M')]
    if not strong_2m.empty:
        pivot_strong = strong_2m.pivot_table(index='Processes', columns='Mode', values='Time')
        # Считаем Speedup для таблицы
        t1_mpi = pivot_strong['MPI'].iloc[0]
        pivot_strong['Speedup_MPI'] = t1_mpi / pivot_strong['MPI']
        print(pivot_strong)
        # Сохраняем в CSV для Excel
        pivot_strong.to_csv('plots/table_strong_2m.csv')
    
    # --- 2. WEAK SCALING EFFICIENCY ---
    print("\n" + "="*50)
    print("TABLE: WEAK SCALING EFFICIENCY (10k per rank)")
    print("="*50)
    weak = df[df['Type'] == 'Weak'].copy()
    if not weak.empty:
        # eff = T(1 rank) / T(P ranks)
        weak['Efficiency'] = weak.groupby('Mode')['Time'].transform(lambda x: x.iloc[0] / x)
        pivot_weak = weak.pivot_table(index='Processes', columns='Mode', values='Efficiency')
        print(pivot_weak)
        # LaTeX
        print("\nLATEX CODE FOR WEAK SCALING:")
        print(pivot_weak.to_latex(float_format="%.2f"))

    # --- plots ---
    print("\nGenerating .png files in /plots folder...")

def generate_strong_plots(df, mode):
    subset = df[(df['Type'] == 'Strong') & (df['Mode'] == mode)].copy()
    if subset.empty: return
    subset = subset.groupby(['Dataset', 'Processes']).agg({'Time': 'mean'}).reset_index()
    unique_datasets = sorted(subset['Dataset'].unique(), key=ds_key)
    
    # Speedup Plot
    plt.figure(figsize=(10, 6))
    for ds in unique_datasets:
        d = subset[subset['Dataset'] == ds].sort_values('Processes')
        t1 = d['Time'].iloc[0]
        plt.plot(d['Processes'], t1 / d['Time'], 'o-', label=f"N = {ds}")
    
    max_p = subset['Processes'].max()
    plt.plot([1, max_p], [1, max_p], 'k--', alpha=0.6, label="Ideal Speedup")
    plt.title(f"Strong Scaling Speedup ({mode})")
    plt.xlabel("Total Cores / Ranks"); plt.ylabel("Speedup")
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.savefig(f"plots/strong_speedup_{mode.lower()}.png")
    plt.close()

print("HPC LOG PARSER v2.0")
df_all = parse_all_logs('.')

if not df_all.empty:
    generate_plots_and_tables(df_all)
    
    for m in ['MPI', 'Hybrid']:
        generate_strong_plots(df_all, m)
    
    print("\nSuccess! All tables printed above and plots saved in 'plots/' directory.")
else:
    print("Error: Could not find any log files. Check your directory structure.")
