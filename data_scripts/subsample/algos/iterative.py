import random

from copy import deepcopy

from utils.progress import track

def iterative_subsample(
    algo: str, comp2ex, ex2comp, comp_sims, template2ex, ex2template, min_freq=1,
    n_trn=1000, ex_sel='random', seed=0, show_progress=True):
    random.seed(seed)
    comp2ex = deepcopy(comp2ex)
    ex2comp = deepcopy(ex2comp)
    for comp, ex_cntr in comp2ex.items():
        for ex, w in ex_cntr.items():
            assert ex in ex2comp and ex2comp[ex][comp] == w
    for ex, comp_cntr in ex2comp.items():
        for comp, w in comp_cntr.items():
            assert comp in comp2ex and comp2ex[comp][ex] == w
    comp2ex = {k: v for k, v in comp2ex.items() if len(v) >= min_freq}
    ex2comp = {ex: {comp: w for comp, w in comp_cntr.items()
                    if comp in comp2ex}
               for ex, comp_cntr in ex2comp.items()}
    for comp, ex_cntr in comp2ex.items():
        for ex, w in ex_cntr.items():
            assert ex in ex2comp and ex2comp[ex][comp] == w
    for ex, comp_cntr in ex2comp.items():
        for comp, w in comp_cntr.items():
            assert comp in comp2ex and comp2ex[comp][ex] == w
    all_comps = list(comp2ex.keys())
    all_comps_set = set(comp2ex.keys())
    comp_wts = {comp: sum(ex_cntr.values()) for comp, ex_cntr in comp2ex.items()}
    comp2idx = {comp: i for i, comp in enumerate(all_comps)}
    trn_comp_freqs = {i: 0 for i, _ in enumerate(all_comps)}
    trn_set = []
    trn_comps = set()
    trn_comps_cycle = set()
    trn_templates_cycle = set()
    for i_ex in track(range(n_trn), disable=not show_progress):
        # select comp
        comp_idxs = [idx for idx, comp in enumerate(all_comps)
                     if comp in comp2ex]
        random.shuffle(comp_idxs)
        if algo == '0' or (algo == '4' and not trn_comps): # random comp
            comp_idx = comp_idxs[0]
        elif algo.startswith('1'): # random unsampled comp
            if algo == '1' and all([idx in trn_comps_cycle for idx in comp_idxs]):
                comp_idx = comp_idxs[0]
            else:
                comp_idx = [idx for idx in comp_idxs
                            if idx not in trn_comps_cycle][0]
        elif algo == '2' or (algo == '24' and not trn_comps): # least freq in sampled
            comp_idx = min(comp_idxs, key=lambda idx: trn_comp_freqs[idx])
        elif algo in ['24', '4'] and trn_comps: # least similar comp
            comp_max_sims = comp_sims[list(trn_comps)].max(axis=0)
            if algo == '24':
                comp_idx = min(comp_idxs,
                               key=lambda idx: (trn_comp_freqs[idx], comp_max_sims[idx]))
            else: # algo == '4'
                comp_idx = min(comp_idxs, key=lambda idx: comp_max_sims[idx])
        elif algo == '25': # (least freq in sampled, most freq in pool) comp
            comp_idx = min(comp_idxs,
                           key=lambda idx: (trn_comp_freqs[idx], -comp_wts[all_comps[idx]]))
        elif algo.startswith('5'): # most freq unsampled comp in pool
            comp_idxs = [idx for idx in comp_idxs if idx not in trn_comps_cycle]
            # comp_idx = min(comp_idxs,
            #                key=lambda idx: -sum(comp2ex[all_comps[idx]].values()))
            comp_idx = min(comp_idxs,
                            key=lambda idx: -comp_wts[all_comps[idx]])
        else:
            raise ValueError(f'Unknown algo: {algo}')

        # select example
        ex = None
        comp = all_comps[comp_idx]
        assert comp in comp2ex and len(comp2ex[comp]) > 0
        exs = list(comp2ex[comp].keys())
        random.shuffle(exs)
        cand_exs = [cand_ex for cand_ex in exs
                    if ex2template[cand_ex] not in trn_templates_cycle]
        if ex_sel == 'random' or ('new_template' in ex_sel  and len(cand_exs) == 0): # random ex
            ex = exs[0]
        elif ex_sel == 'new_template': # ex with new template
            ex = cand_exs[0]
        elif ex_sel == 'mf_new_template': # ex with most freq new template
            ex = max(cand_exs, key=lambda ex: len(template2ex[ex2template[ex]]))
        else:
            raise ValueError('Not implemented')

        if ex is None:
            raise Exception('No compounds with examples')

        trn_set.append(ex)
        # NOTE: no need to delete ex from template2ex for the new template as it
        # will not be considered till next cycle and since there's only one template
        # per example, the ranking of templates will be the same even if we did.
        trn_templates_cycle.add(ex2template[ex])
        if all([template in trn_templates_cycle for template in template2ex]):
            trn_templates_cycle.clear()

        # update graph
        for comp, w in ex2comp[ex].items():
            # if comp not in all_comps_set: continue
            if comp not in comp2ex: continue
            del comp2ex[comp][ex]
            comp_wts[comp] -= w
            assert comp_wts[comp] >= 0
            trn_comp_freqs[comp2idx[comp]] += w
            trn_comps.add(comp2idx[comp])
            trn_comps_cycle.add(comp2idx[comp])
            if len(comp2ex[comp]) == 0:
                del comp2ex[comp]
                del comp_wts[comp]
        del ex2comp[ex]

        if '_cyc' in algo and all([comp2idx[comp] in trn_comps_cycle
                                   for comp in comp2ex]): # reset if everything samplesd
            trn_comps_cycle.clear()

    print(f'{len(trn_comps)} of {len(all_comps)} compounds with min_freq >= {min_freq} sampled')
    return trn_set
