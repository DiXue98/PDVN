from __future__ import print_function
import numpy as np
import torch
import torch.nn.functional as F
from rdkit import Chem
import rdchiral
from rdchiral.main import rdchiralRunText, rdchiralRun
from rdchiral.initialization import rdchiralReaction, rdchiralReactants
from .mlp_policies import load_model, load_parallel_model , preprocess
from collections import defaultdict, OrderedDict

def merge(reactant_d):
    ret = []
    for reactant, l in reactant_d.items():
        ss, srs, ts, ids = zip(*l)
    #     k = np.argmax(ss)
    #     ret.append((reactant, sum(ss), list(ts)[k], list(ids)[k]))
    # reactants, scores, templates, templates_idx = zip(*sorted(ret, key=lambda item : item[1], reverse=True))
        ret.append((reactant, sum(ss), sum(srs), list(ts)[0], list(ids)[0]))
    reactants, scores, scores_reference, templates, templates_idx = zip(*sorted(ret, key=lambda item : item[1], reverse=True))
    return list(reactants), list(scores), list(scores_reference), list(templates), list(templates_idx)



class MLPModel(object):
    def __init__(self, state_path, template_path, device=-1, fp_dim=2048, realistic_filter=False):
        super(MLPModel, self).__init__()
        self.fp_dim = fp_dim
        try:
            self.net, self.idx2rules = load_parallel_model(state_path, template_path, fp_dim)
        except:
            self.net, self.idx2rules = load_model(state_path, template_path, fp_dim)
        self.net.eval()
        self.device = device
        if device >= 0:
            self.net.to(device)

        self.realistic_filter = realistic_filter

        self.reference_net, _ = load_parallel_model('one_step_model/saved_rollout_state_1_2048.ckpt', template_path, fp_dim)
        self.reference_net.eval()
        if device >= 0:
            self.reference_net.to(device)

    def forward_topk(self, arr, topk=10):
        preds = self.net(arr)
        preds_reference = self.reference_net(arr).detach()

        if self.realistic_filter:
            preds_reference, idx_topk = torch.topk(preds_reference, k=topk)
            preds = preds.gather(1, idx_topk)
        else:
            preds, idx_topk = torch.topk(preds, k=topk)
            preds_reference = preds_reference.gather(1, idx_topk)

        return preds, preds_reference, idx_topk

    def run(self, mol_node, topk=10, backward=True, sample_mode=None, test=False):
        x = mol_node.mol if hasattr(mol_node, 'mol') else mol_node
        arr = preprocess(x, self.fp_dim)
        arr = np.reshape(arr,[-1, arr.shape[0]])
        arr = torch.tensor(arr, dtype=torch.float32)
        if self.device >= 0:
            arr = arr.to(self.device)
        # preds = self.net(arr)
        # preds = F.softmax(preds,dim=1)
        # if self.device >= 0:
        #     preds = preds.cpu()
        # probs, idx = torch.topk(preds,k=topk)
        preds, preds_reference, idx = self.forward_topk(arr, topk=topk)
        probs = F.softmax(preds, dim=1)
        probs_reference = F.softmax(preds_reference, dim=1)
        preds, preds_reference, idx, probs, probs_reference = preds.cpu(), preds_reference.cpu(), idx.cpu(), probs.cpu(), probs_reference.cpu()
        # probs = F.softmax(preds, dim=1)
        # probs_reference = F.softmax(preds_reference, dim=1)
        rule_k = [self.idx2rules[id] for id in idx[0].numpy().tolist()]
        reactants = []
        scores = []
        scores_reference = []
        templates = []
        templates_idx = []

        if sample_mode == 'template':
            # probs = probs / probs.sum()
            result = {'reactants':[],
                      'reactant_lists': [],
                      'template' : [],
                      'templates_idx': []}
            ancestors = mol_node.get_ancestors()
            for i in range(topk):
                try:
                    is_invalid = True # if current template, i.e., idx[i], is valid
                    if test:
                        template_idx = torch.argmax(probs[0]).item()
                    else:
                        template_idx = torch.multinomial(probs[0], 1).item() # TODO: sum of probabilities <= 0
                    rule = rule_k[template_idx]
                    out1 = rdchiralRunText(rule, x)
                    out1 = sorted(out1)
                    # check repeated molecules
                    for reactants in out1:
                        reactant_list = list(set(reactants.split('.')))
                        is_repeated = False
                        # for reactant in reactant_list:
                        #     if reactant in ancestors:
                        #         is_repeated = True
                        #         break
                        if not is_repeated:
                            is_invalid = False
                            result['reactants'].append(reactants)
                            result['reactant_lists'].append(reactant_list)
                            result['template'].append(rule)
                            result['templates_idx'].append(template_idx if self.realistic_filter else idx[0][template_idx].item())
                except (ValueError, RuntimeError) as e:
                    """
                    RuntimeError: Pre-condition Violation
                    Stereo atoms should be specified before specifying CIS/TRANS bond stereochemistry
                    Violation occurred on line 288 in file Code/GraphMol/Bond.h
                    Failed Expression: what <= STEREOE || getStereoAtoms().size() == 2
                    RDKIT: 2020.09.1
                    BOOST: 1_73
                    """
                    pass
                except (IndexError, KeyError) as e:
                    """
                    rdchiral bug during function call rdchiralRunText(rule, mol)
                    This error can be reprobuced by the following code:
                    mol = 'C[C@H](OC(=O)C=O)C(=O)O'
                    rule = '([#8:1]-[C:2](=[O;D1;H0:3])-[CH;D2;+0:4]=[O;H0;D1;+0:5])>>[#8:1]-[C:2](=[O;D1;H0:3])-[C@@H;D3;+0:4](-[OH;D1;+0:5])-[C@H;D3;+0:4](-[OH;D1;+0:5])-[C:2](-[#8:1])=[O;D1;H0:3]'
                    out1 = rdchiralRunText(rule, mol)
                    """
                    pass
                if is_invalid: # invalid molecule
                    result['reactants'].append('Invalid')
                    result['reactant_lists'].append(['Invalid'])
                    result['template'].append(rule)
                    result['templates_idx'].append(template_idx if self.realistic_filter else idx[0][template_idx].item())
                    probs[0][template_idx] = 0
                else:
                    return result

            return None
            
        for i , rule in enumerate(rule_k):
            out1 = []
            try:
                if backward:
                    out1 = rdchiralRunText(rule, x)
                else:
                    rxn_prod, rxn_agent, rxn_react = rule.split(">")
                    reversed_rule = '(' + rxn_react + ')>' + rxn_agent + '>' + rxn_prod[1:-1]
                    out1 = rdchiralRunText(reversed_rule, x)
                # out1 = rdchiralRunText(rule, Chem.MolToSmiles(Chem.MolFromSmarts(x)))
                if len(out1) == 0: continue
                # if len(out1) > 1: print("more than two reactants."),print(out1)
                out1 = sorted(out1)
                for reactant in out1:
                    reactants.append(reactant)
                    scores.append(probs[0][i].item()/len(out1))
                    scores_reference.append(probs_reference[0][i].item()/len(out1))
                    templates.append(rule)
                    templates_idx.append(i if self.realistic_filter else idx[0][i].item())
            # out1 = rdchiralRunText(x, rule)
            except (ValueError, RuntimeError) as e:
                """
                RuntimeError: Pre-condition Violation
                Stereo atoms should be specified before specifying CIS/TRANS bond stereochemistry
                Violation occurred on line 288 in file Code/GraphMol/Bond.h
                Failed Expression: what <= STEREOE || getStereoAtoms().size() == 2
                RDKIT: 2020.09.1
                BOOST: 1_73
                """
                pass
            except (IndexError, KeyError) as e:
                """
                rdchiral bug during function call rdchiralRunText(rule, mol)
                This error can be reprobuced by the following code:
                mol = 'C[C@H](OC(=O)C=O)C(=O)O'
                rule = '([#8:1]-[C:2](=[O;D1;H0:3])-[CH;D2;+0:4]=[O;H0;D1;+0:5])>>[#8:1]-[C:2](=[O;D1;H0:3])-[C@@H;D3;+0:4](-[OH;D1;+0:5])-[C@H;D3;+0:4](-[OH;D1;+0:5])-[C:2](-[#8:1])=[O;D1;H0:3]'
                out1 = rdchiralRunText(rule, mol)
                """
                pass
        if len(reactants) == 0: return None
        reactants_d = defaultdict(list)
        for r, s, sr, t, id in zip(reactants, scores, scores_reference, templates, templates_idx):
            if '.' in r:
                str_list = sorted(r.strip().split('.'))
                reactants_d['.'.join(str_list)].append((s, sr, t, id))
            else:
                reactants_d[r].append((s, sr, t, id))

        reactants, scores, scores_reference, templates, templates_idx = merge(reactants_d)
        total = sum(scores)
        total_reference = sum(scores_reference)
        scores = [s / total for s in scores]
        scores_reference = [s / total_reference for s in scores_reference]

        if sample_mode == 'molecule':
            reactant_idx = np.random.choice(len(scores), p=scores)
            return {'reactants':reactants[reactant_idx:reactant_idx+1],
                    'scores' : scores[reactant_idx:reactant_idx+1],
                    'template' : templates[reactant_idx:reactant_idx+1]}

        return {'reactants':reactants,
                'scores' : scores,
                'scores_reference' : scores_reference,
                'template' : templates,
                'templates_idx': templates_idx}



if __name__ == '__main__':
    import argparse
    from pprint import pprint
    parser = argparse.ArgumentParser(description="Policies for retrosynthesis Planner")
    parser.add_argument('--template_rule_path', default='../data/uspto_all/template_rules_1.dat',
                        type=str, help='Specify the path of all template rules.')
    parser.add_argument('--model_path', default='../model/saved_rollout_state_1_2048.ckpt',
                        type=str, help='specify where the trained model is')
    args = parser.parse_args()
    state_path = args.model_path
    template_path = args.template_rule_path
    model =  MLPModel(state_path,template_path,device=-1)
    x = '[F-:1]'
    # x = '[CH2:10]([S:14]([O:3][CH2:2][CH2:1][Cl:4])(=[O:16])=[O:15])[CH:11]([CH3:13])[CH3:12]'
    # x = '[S:3](=[O:4])(=[O:5])([O:6][CH2:7][CH:8]([CH2:9][CH2:10][CH2:11][CH3:12])[CH2:13][CH3:14])[OH:15]'
    # x = 'OCC(=O)OCCCO'
    # x = 'CC(=O)NC1=CC=C(O)C=C1'
    x = 'S=C(Cl)(Cl)'
    # x = "NCCNC(=O)c1ccc(/C=N/Nc2ncnc3c2cnn3-c2ccccc2)cc1"
    # x = 'CCOC(=O)c1cnc2c(F)cc(Br)cc2c1O'
    # x = 'COc1cc2ncnc(Oc3cc(NC(=O)Nc4cc(C(C)(C)C(F)(F)F)on4)ccc3F)c2cc1OC'
    # x = 'COC(=O)c1ccc(CN2C(=O)C3(COc4cc5c(cc43)OCCO5)c3ccccc32)o1'
    x = 'O=C1Nc2ccccc2C12COc1cc3c(cc12)OCCO3'
    # x = 'CO[C@H](CC(=O)O)C(=O)O'
    # x = 'O=C(O)c1cc(OCC(F)(F)F)c(C2CC2)cn1'
    y = model.run(x,10)
    pprint(y)
