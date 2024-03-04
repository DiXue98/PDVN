# Simple route visualizer

Turns a single route json into a pdf. Example command:

```bash
# Note: will save "tree1.pdf"
python visualize_tree.py --input_file sample_route_1.json --output_file ./tree1
```

How to read the tree:

- Ovular nodes are molecules, square nodes are reactions.
- outline colour is whether it is `in_stock` or not: red means no, green means yes.
- Each molecule is labelled by its smiles. You could also add addition data, e.g. cost. I put this into the script on line 96
- Reactions are currently just labelled by the template hash, but this could easily be changed (modify line 115)