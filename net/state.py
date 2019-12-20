

# get whether this module and each submodule recursively is in train or evaluation mode
def get_train_mode_tree(module):
    def _get_mode(module, mode):
        mode.append([module.training])
        for submodule in module.children():
            mode[-1].append(_get_mode(submodule, mode))
        return mode
    return _get_mode(module, [])


# set the train or evaluation state of this module and each submodule recursively
def set_train_mode_tree(module, mode):
    module.train(mode[0])
    for s, submodule in enumerate(module.children()):
        set_train_mode_tree(submodule, mode[s])
