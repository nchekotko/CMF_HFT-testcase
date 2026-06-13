"""Dump text outputs of each cell, skip image data."""
import json, sys
nb = json.load(open("week1_exploration.ipynb", encoding="utf-8"))
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] != "code":
        continue
    out = c.get("outputs", [])
    if not out:
        continue
    print(f"\n========== cell {i} (exec {c.get('execution_count')}) ==========")
    for o in out:
        t = o.get("output_type")
        if t == "stream":
            txt = "".join(o.get("text", []))
            print(txt, end="")
        elif t == "execute_result":
            data = o.get("data", {})
            txt = data.get("text/plain", [])
            if isinstance(txt, list): txt = "".join(txt)
            print(txt)
        elif t == "display_data":
            data = o.get("data", {})
            if "image/png" in data:
                print("[image/png]")
            elif "text/plain" in data:
                txt = data["text/plain"]
                if isinstance(txt, list): txt = "".join(txt)
                print(txt)
        elif t == "error":
            print("ERROR:", o.get("ename"), o.get("evalue"))
