from .market1501 import Market1501


__factory = {
        'market1501': Market1501,
    }


dataset_shortcut = {
    'market1501': 'M',
}


def create_dataset(cfg, samples=None):
    return __factory[cfg.NAME](cfg, samples=samples)
