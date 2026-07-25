import os
import json

leaderboard_file = r"D:\trading strats\leaderboard.json"
output_dir = r"D:\trading strats"

if os.path.exists(leaderboard_file):
    with open(leaderboard_file, "r") as f:
        data = json.load(f)
        
    # Deduplicate keeping best score
    seen = {}
    for entry in data:
        strat_id = entry["id"]
        score = entry.get("score", entry.get("sharpe", 0))
        if strat_id not in seen or score > seen[strat_id].get("score", seen[strat_id].get("sharpe", 0)):
            seen[strat_id] = entry
            
    unique_entries = list(seen.values())
    # Sort descending by score
    unique_entries.sort(key=lambda x: x.get("score", x["sharpe"]), reverse=True)
    
    # Take top 10
    top_10 = unique_entries[:10]
    top_10_ids = set(x["id"] for x in top_10)
    
    # Save back
    with open(leaderboard_file, "w") as f:
        json.dump(top_10, f, indent=4)
        
    print(f"[+] Deduplicated and saved top 10 entries to: {leaderboard_file}")
    
    # Clean up obsolete C++ files in output directory
    for file in os.listdir(output_dir):
        if file.endswith(".cpp"):
            strat_id = file[:-4]
            if strat_id not in top_10_ids:
                try:
                    os.remove(os.path.join(output_dir, file))
                    print(f"    [-] Removed obsolete C++ strategy file: {file}")
                except Exception as e:
                    print(f"    [!] Failed to remove {file}: {e}")
else:
    print("[!] Leaderboard file not found.")
