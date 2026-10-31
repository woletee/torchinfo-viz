from __future__ import annotations


def insert_containers(layers, index, layer, hierarchy, current, layer_ids):
    offset = 0
    for depth in range(1, layer.depth):
        parent = hierarchy[depth]
        if (
            depth not in current or current[depth].module is not parent.module
        ) and parent is not layers[index + offset - 1]:
            parent.calculate_num_params()
            parent.check_recursive(layer_ids)
            layers.insert(index + offset, parent)
            layer_ids.add(parent.layer_id)
            current[depth] = parent
            offset += 1
