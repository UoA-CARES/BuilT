import pkgutil
import sys
import os


def load_all_modules_from_dir(dirname):
    for importer, package_name, _ in pkgutil.iter_modules([dirname]):
        if package_name not in sys.modules and package_name != 'main':
            module = importer.find_module(package_name).load_module(package_name)
            print(f'{module} is loaded')

print(os.path.dirname(__file__))
load_all_modules_from_dir(os.path.dirname(__file__))