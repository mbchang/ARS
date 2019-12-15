# Code in this file is copied and adapted from
# https://github.com/openai/evolution-strategies-starter.

import numpy as np

def itergroups(items, group_size):
    assert group_size >= 1
    group = []
    for x in items:
        group.append(x)
        if len(group) == group_size:
            yield tuple(group)
            del group[:]
    if group:
        yield tuple(group)



def batched_weighted_sum(weights, vecs, batch_size):
    total = 0
    num_items_summed = 0
    for batch_weights, batch_vecs in zip(itergroups(weights, batch_size),
                                         itergroups(vecs, batch_size)):
        assert len(batch_weights) == len(batch_vecs) <= batch_size
        total += np.dot(np.asarray(batch_weights, dtype=np.float64),
                        np.asarray(batch_vecs, dtype=np.float64))
        num_items_summed += len(batch_weights)
    return total, num_items_summed

def tranpose_list_of_dicts(list_of_dicts):
    """
    input:
        [
            {key1: val1_0, key2: val2_0, ...},
            {key1: val1_1, key2: val2_1, ...},
        ]

    output:
        {
            key1: [val1_0, val1_1, ...]
            key2: [val2_0, val2_1, ...]
        }
    """
    keys = list_of_dicts[0].keys()
    output = {key: [] for key in keys}
    for element_dict in list_of_dicts:
        for key in element_dict:
            output[key].append(element_dict[key])
    return output
