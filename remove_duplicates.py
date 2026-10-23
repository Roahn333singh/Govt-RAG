import os
import hashlib
from collections import defaultdict

def get_file_hash(filepath, block_size=65536):
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for block in iter(lambda: f.read(block_size), b''):
            hasher.update(block)
    return hasher.hexdigest()

def remove_duplicates(root_dir):
    print(f"Scanning directory: {root_dir}")
    
    # Step 1: Group files by size to avoid hashing everything
    size_dict = defaultdict(list)
    total_files = 0
    
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            # Skip symlinks
            if os.path.islink(filepath):
                continue
            
            try:
                file_size = os.path.getsize(filepath)
                size_dict[file_size].append(filepath)
                total_files += 1
            except OSError as e:
                print(f"Error accessing {filepath}: {e}")

    print(f"Total files scanned: {total_files}")
    
    # Step 2: Hash files that share the same size and remove duplicates
    hash_dict = {}
    duplicates_removed = 0
    bytes_freed = 0
    
    for size, filepaths in size_dict.items():
        if len(filepaths) > 1:
            for filepath in filepaths:
                try:
                    file_hash = get_file_hash(filepath)
                    if file_hash in hash_dict:
                        # It's a duplicate, delete it
                        print(f"Removing duplicate: {filepath}")
                        os.remove(filepath)
                        duplicates_removed += 1
                        bytes_freed += size
                    else:
                        hash_dict[file_hash] = filepath
                except OSError as e:
                    print(f"Error hashing/deleting {filepath}: {e}")
                    
    print("\n--- Summary ---")
    print(f"Total files scanned: {total_files}")
    print(f"Duplicates removed: {duplicates_removed}")
    print(f"Storage freed: {bytes_freed / (1024*1024):.2f} MB")

if __name__ == "__main__":
    target_directory = "rag-data"
    if os.path.exists(target_directory):
        remove_duplicates(target_directory)
    else:
        print(f"Directory '{target_directory}' not found.")
