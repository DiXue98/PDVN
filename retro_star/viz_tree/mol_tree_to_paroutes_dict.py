from retro_star.alg.mol_node import MolNode
from retro_star.alg.reaction_node import ReactionNode

from collections import defaultdict

def mol_tree_to_paroutes_dict(root_node):
    # Populate basic dict fields
    out = defaultdict(list)
    if isinstance(root_node, MolNode):        
        out["smiles"] = root_node.mol
        out["type"] = "mol"
        out["in_stock"] = root_node.is_known
        out["value_reference"] = 0
        out["succ_value"] = root_node.succ_value
    elif isinstance(root_node, ReactionNode):
        out["type"] = "reaction"
        out["cost"] = root_node.cost
        out["cost_reference"] = root_node.cost_reference
        out["value_reference"] = root_node.cost_reference
        out["template_rule"] = root_node.template
        out["succ_value"] = root_node.succ_value
    else:
        raise ValueError

    # Add children (recursive call)
    if len(root_node.children) > 0:
        if isinstance(root_node, ReactionNode):
            for child in root_node.children:
                assert child.succ
                child_dict = mol_tree_to_paroutes_dict(child)
                out["children"].append(child_dict)
                out["value_reference"] += child_dict["value_reference"]

        elif isinstance(root_node, MolNode):
            best_child = None
            for child in root_node.children:
                if (child.succ and (best_child is None or
                                    child.succ_value < best_child.succ_value)):
                    best_child = child
            assert root_node.succ_value == best_child.succ_value, f"{root_node.succ_value} != {best_child.succ_value}"

            child_dict = mol_tree_to_paroutes_dict(best_child)
            out["children"].append(child_dict)
            out["value_reference"] = child_dict["value_reference"]

    return out