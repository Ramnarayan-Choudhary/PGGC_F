import subprocess
import json
import os
import time
import argparse

def get_total_items(input_file):
    if input_file.endswith(".jsonl"):
        with open(input_file, 'r') as f:
            return sum(1 for _ in f)
    else:
        with open(input_file, 'r') as f:
            data = json.load(f)
            return len(data)

def main():
    parser = argparse.ArgumentParser(description="Parallel Launcher")
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--num_gpus", type=int, default=4, help="Number of GPUs to split work across")
    parser.add_argument("--iterations", type=int, default=25)
    parser.add_argument("--base_model", type=str, default="/home/ramnarayan.ramniwas/.cache/huggingface/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/8afb486c1db24fe5011ec46dfbe5b5dccdb575c2")
    parser.add_argument("--hf_home", type=str, default="/home/ramnarayan.ramniwas/.cache/huggingface")
    
    args = parser.parse_args()
    
    total_items = get_total_items(args.input_file)
    chunk_size = total_items // args.num_gpus
    
    processes = []
    
    print(f"Total Items: {total_items}")
    print(f"Splitting across {args.num_gpus} GPUs. Chunk size: ~{chunk_size}")
    
    # Launch Loop
    for i in range(args.num_gpus):
        start = i * chunk_size
        # Ensure last process gets the remainder
        end = (i + 1) * chunk_size if i < args.num_gpus - 1 else total_items
        
        output_file = f"results_part_{i}.json"
        log_file = f"log_part_{i}.txt"
        
        # Construct Environment
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(i) # Assign specific GPU
        env["HF_HOME"] = args.hf_home
        
        # Construct Command
        cmd = [
            "python3", "scripts/run_defense.py",
            "--input_file", args.input_file,
            "--output_file", output_file,
            "--start_index", str(start),
            "--end_index", str(end),
            "--iterations", str(args.iterations),
            "--model", args.base_model
        ]
        
        print(f"Launching Worker {i} (Items {start}-{end}) on GPU {i} -> Logs: {log_file}")
        
        with open(log_file, "w") as log_f:
            p = subprocess.Popen(cmd, env=env, stdout=log_f, stderr=log_f)
            processes.append(p)
            
    print("All workers launched. Monitoring...")
    
    try:
        while True:
            alive = sum(1 for p in processes if p.poll() is None)
            if alive == 0:
                print("All processes completed.")
                break
            print(f"Workers running: {alive}/{args.num_gpus}. Waiting...")
            time.sleep(30)
    except KeyboardInterrupt:
        print("Terminating all workers...")
        for p in processes:
            p.terminate()
            
    # Merge Results
    print("Merging results...")
    final_results = []
    for i in range(args.num_gpus):
        fname = f"results_part_{i}.json"
        if os.path.exists(fname):
            with open(fname, 'r') as f:
                try:
                    part_data = json.load(f)
                    final_results.extend(part_data)
                except json.JSONDecodeError:
                    print(f"Error reading {fname}")
        else:
            print(f"Missing file: {fname}")
            
    with open("results_final_merged.json", "w") as f:
        json.dump(final_results, f, indent=4)
        
    print(f"Merged {len(final_results)} items into results_final_merged.json")

if __name__ == "__main__":
    main()
