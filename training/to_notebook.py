"""Convertit train_melanoscan.py (cellules « # %% ») en notebook Jupyter."""
import os
import re
import sys

import nbformat

src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "train_melanoscan.py")
text = open(src, encoding="utf-8").read()
cells = []
for chunk in re.split(r"^# %%", text, flags=re.M)[1:]:
    header, _, body = chunk.partition("\n")
    body = body.strip("\n")
    if header.strip() == "[markdown]":
        cells.append(nbformat.v4.new_markdown_cell(re.sub(r"^# ?", "", body, flags=re.M)))
    else:
        cells.append(nbformat.v4.new_code_cell(body))
nb = nbformat.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbformat.write(nb, src.replace(".py", ".ipynb"))
