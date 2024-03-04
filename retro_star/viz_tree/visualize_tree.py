"""Standalone script to visualize a tree in PaRoutes json format."""

import argparse
import json
import tempfile
import collections
from pathlib import Path
from typing import Any, Optional, Dict, List, Deque

from PIL.Image import Image as PilImage
from PIL import Image
from graphviz import Digraph
from rdkit import Chem
from rdkit.Chem import Draw
import numpy as np


def az_crop_image(img: PilImage, margin: int = 20) -> PilImage:
    """
    Crop an image by removing white space around it

    :param img: the image to crop
    :param margin: padding, defaults to 20
    :return: the cropped image

    NOTE: this function was directly copied from AiZynthfinder
    """
    # pylint: disable=invalid-name
    # First find the boundaries of the white area
    x0_lim = img.width
    y0_lim = img.height
    x1_lim = 0
    y1_lim = 0
    for x in range(0, img.width):
        for y in range(0, img.height):
            if img.getpixel((x, y)) != (255, 255, 255):
                if x < x0_lim:
                    x0_lim = x
                if x > x1_lim:
                    x1_lim = x
                if y < y0_lim:
                    y0_lim = y
                if y > y1_lim:
                    y1_lim = y
    x0_lim = max(x0_lim, 0)
    y0_lim = max(y0_lim, 0)
    x1_lim = min(x1_lim + 1, img.width)
    y1_lim = min(y1_lim + 1, img.height)
    # Then crop to this area
    cropped = img.crop((x0_lim, y0_lim, x1_lim, y1_lim))
    # Then create a new image with the desired padding
    out = Image.new(
        img.mode,
        (cropped.width + 2 * margin, cropped.height + 2 * margin),
        color="white",
    )
    out.paste(cropped, (margin + 1, margin + 1))
    return out


def basic_viz_dict(
    tree: Dict[str, Any],
    filename: str,
    max_nodes: int = 100,
    title: Optional[str] = None,
    draw_mols: bool = True,
) -> None:

    # Explicitly get all the nodes
    nodes: List[Dict[str, Any]] = []
    node_queue: Deque[Dict[str, Any]] = collections.deque([tree])
    while len(node_queue) > 0:
        n = node_queue.popleft()
        for child in n.get("children", []):
            node_queue.append(child)
        nodes.append(n)

    # Check that there aren't too many nodes
    # Otherwise graphviz will be super slow
    assert len(nodes) <= max_nodes, "Visualization will be too slow. Pass in a higher value of max_nodes to override this."

    # Init graph
    G = Digraph("G", filename=filename)
    G.format = "pdf"

    # Node names
    node_names = []
    temp_files = []
    for idx, node in enumerate(nodes):

        # Names and node colours
        image_file = None
        if node.get("type") == "mol":
            name = node["smiles"]
            # TODO: add stuff to "name" if desired here,
            # e.g.:
            # name += f"\ncost=???"
            if node.get("in_stock"):
                colour = "green"
            else:
                colour = "red"
            shape = "ellipse"

            if draw_mols:
                mol_obj = Chem.MolFromSmiles(node["smiles"])
                if mol_obj is not None:
                    img_obj = az_crop_image(Draw.MolToImage(mol_obj))
                    _, temp_file_path = tempfile.mkstemp(suffix=".png")
                    img_obj.save(temp_file_path)
                    temp_files.append(temp_file_path)
                    image_file = temp_file_path

        elif node.get("type") == "reaction":
            name = ""
            # TODO: add reaction info here, e.g.
            # name += f"template hash: {node.get('metadata', dict()).get('template_hash')}"
            name += f"template rule: {node.get('template_rule')}"
            neg_log_p = node.get('cost')
            name += f"\np= {np.exp(-neg_log_p)}"
            name += f"\n-logp= {neg_log_p}"
            succ_value = node.get('succ_value')
            name += f"\nsucc_value= {succ_value}"
            neg_log_p_reference = node.get('cost_reference')
            name += f"\n-logp (reference)= {neg_log_p_reference}"
            value_reference = node.get('value_reference')
            name += f"\nvalue (reference)= {value_reference}"
            colour = "black"
            shape = "box"
        else:
            raise ValueError
        node_names.append(name)

        # Fill colours
        fill_color = "white"

        # Make node
        if image_file is None:
            label = name
        else:
            name_no_newlines = name.replace("\n", "<BR/>")
            label = f"""<
<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0">
  <TR>
    <TD><IMG SRC="{image_file}" SCALE="TRUE"/></TD>
  </TR>
  <TR>
    <TD>{name_no_newlines}</TD>
  </TR>
</TABLE>>"""
        G.node(
            str(idx),
            label=label,
            fontsize="10.0",
            shape=shape,
            color=colour,
            style="filled",
            fillcolor=fill_color,
        )

    # Add edges:
    for idx, node in enumerate(nodes):
        for child in node.get("children", []):

            # Only visualize an edge if the parent is also being visualized
            child_idx = None
            for idx2, n in enumerate(nodes):
                if child is n:
                    child_idx = idx2
                    G.edge(str(idx), str(child_idx), label="")
                    break

    # Add overall title
    if title is not None:
        G.attr(label=title)

    G.render()

    # Remove temp file
    path = Path(filename)
    if path.exists() and path.is_file():
        path.unlink()

    # Remove temp mol files
    for tmp_file in temp_files:
        tmp_file_path = Path(tmp_file)
        if tmp_file_path.exists():
            tmp_file_path.unlink()


def main(input_file: str, output_file: str):
    with open(input_file) as f:
        route_dict = json.load(f)
    basic_viz_dict(
        tree=route_dict,
        filename=output_file,
        draw_mols=True,
        title=f"Visualization of {input_file}",  # TODO: change this to whatever you want
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--output_file", type=str)
    args = parser.parse_args()
    main(input_file=args.input_file, output_file=args.output_file)