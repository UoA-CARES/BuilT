from __future__ import print_function

from easydict import EasyDict as edict


class Category:
    def __init__(self, name):
        self._name = name
        self._class_dict = dict()

    def __repr__(self):
        format_str = self.__class__.__name__
        format_str += f'(name={self._name}, items={list(self._class_dict.keys())})'
        return format_str

    @property
    def name(self):
        return self._name

    @property
    def class_dict(self):
        return self._class_dict

    def get(self, key: str):
        return self._class_dict.get(key, None)

    def add(self, clazz):
        """add a callable class.

        Args:
            clazz: callable class to be registered
        """
        if not callable(clazz):
            raise ValueError(f'object must be callable')

        class_name = clazz.__name__
        if class_name in self._class_dict:            
            raise KeyError(f'{class_name} is already registered in {self.name}')

        self._class_dict[class_name] = clazz
        return clazz


class Registry:
    def __init__(self):
        self.categories = edict()

    def __repr__(self):
        format_str = self.__class__.__name__
        for k, v in self.categories.items():
            format_str += f'(category[{k}]: {v})'

        return format_str        

    def add(self, category, clazz=None):
        self.categories[category] = Category(category)
        if clazz is not None:
            self.categories[category].add(clazz)

    def build_from_config(self, category, config, default_args=None):
        """Build a callable object from configuation dict.

        Args:
            config (dict): Configuration dict. It should contain the key "name".
            category (:obj:`Registry`): The registry to search the name from.
            default_args (dict, optional): Default initialization argments.
        """
        assert isinstance(config, dict) and 'name' in config
        assert isinstance(default_args, dict) or default_args is None
        
        name = config['name']
        name = name.replace('-', '_')
        clazz = self.categories.category.get(name)
        if clazz is None:
            raise KeyError(f'{name} is not in the {category} registry')

        args = dict()
        if default_args is not None:
            args.update(default_args)
        if 'params' in config:
            args.update(config['params'])
        return clazz(**args)
