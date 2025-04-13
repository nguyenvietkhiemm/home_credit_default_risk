import os
import glob
import subprocess
from pathlib import Path
from sitecustomize import ROOT

folder = ROOT + "/jupyter"
prefixes = ['000', '002', '102', '202', '302', '402', '502', '602']
output_folder = ROOT + "/src/engineering"

def clean_and_align_script(script_path):
    with open(script_path, "r", encoding="utf-8") as file:
        lines = file.readlines()
    cleaned_lines = [line for line in lines if line.strip() and not line.strip().startswith("#")]
    with open(script_path, "w", encoding="utf-8") as file:
        file.writelines(cleaned_lines)


for prefix in prefixes:
    pattern = str(folder + f"/{prefix}_*.ipynb")
    for notebook_path in glob.glob(pattern):
        notebook_path = Path(notebook_path)
        print(f"Converting: {notebook_path.name}")

        subprocess.run([
            "jupyter", "nbconvert",
            "--to", "script",
            "--output-dir", str(output_folder),
            str(notebook_path)
        ])

        script_path = Path(output_folder) / f"{notebook_path.stem}.py"
        clean_and_align_script(script_path)
