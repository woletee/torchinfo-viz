from __future__ import annotations


def sparse_members(data):
    members = [data.values()]
    for name in (
        "indices",
        "crow_indices",
        "col_indices",
        "ccol_indices",
        "row_indices",
    ):
        getter = getattr(data, name, None)
        if getter is None:
            continue
        try:
            member = getter()
        except (NotImplementedError, RuntimeError):
            continue
        members.append(member)
    return members
