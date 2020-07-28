from __future__ import print_function

from singleton_decorator import SingletonDecorator


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
            raise KeyError(
                f'{class_name} is already registered in {self.name}')

        self._class_dict[class_name] = clazz
        return clazz


@SingletonDecorator
class Registry:
    def __init__(self):
        self.categories = dict()

    def __repr__(self):
        format_str = self.__class__.__name__
        for k, v in self.categories.items():
            format_str += f'(category[{k}]: {v})'

        return format_str

    def clear(self):
        self.categories.clear()

    def add(self, category, clazz=None):
        if category not in self.categories:
            self.categories[category] = Category(category)

        if clazz is not None:
            self.categories[category].add(clazz)

    def build_from_config(self, category, config, default_args=None):
        """Build a callable object from configuation dict.

        Args:
            category: The name of category to search the name from.
            config (dict): Configuration dict. It should contain the key "name".            
            default_args (dict, optional): Default initialization argments.
        """
        assert isinstance(config, dict) and 'name' in config
        assert isinstance(default_args, dict) or default_args is None

        name = config['name']
        name = name.replace('-', '_')
        clazz = self.categories[category].get(name)
        if clazz is None:
            raise KeyError(f'{name} is not in the {category} registry')

        args = dict()
        if default_args is not None:
            args.update(default_args)
        if 'params' in config:
            args.update(config['params'])
        return clazz(**args)

    def build_model(self, config, **kwargs):
        return self.build_from_config('model', config.model, kwargs)

    def build_loss_fn(self, config, **kwargs):
        return self.build_from_config('loss', config.loss, kwargs)

    def build_optimizer(self, config, **kwargs):
        return self.build_from_config('optimizer', config.optimizer, kwargs)

    def build_scheduler(self, config, **kwargs):
        return self.build_from_config('scheduler', config.scheduler, kwargs)

    def build_hooks(self):
        pass
    
    def build_dataloaders(self):
        pass


registry = Registry()
