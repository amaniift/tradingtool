with open("src/data/data_manager.py", "r") as f:
    lines = f.readlines()

start_idx = -1
for i, line in enumerate(lines):
    if "if not payload:" in line and "import datetime, math" in lines[i+1]:
        start_idx = i
        break

end_idx = -1
for i, line in enumerate(lines):
    if "return {" in line and '"chain_df": chain_df,' in lines[i+5] and start_idx != -1 and i > start_idx:
        end_idx = i + 7 
        break

# verify end_idx covers the return properly
if start_idx != -1 and end_idx != -1:
    del lines[start_idx:end_idx]
    with open("src/data/data_manager.py", "w") as f:
        f.writelines(lines)
    print(f"Removed {end_idx - start_idx} lines of old payload fallback code. Scope shadow fixed.")
else:
    print("Failed to find boundaries.")
