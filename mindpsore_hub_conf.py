"""hub config."""
from src.resnet import resnet20


def create_network(name, *args, **kwargs):
    """ create network """
    if name == 'resnet20':
        return resnet20(*args, **kwargs)
    raise NotImplementedError("{name} is not implemented in the repo")
    