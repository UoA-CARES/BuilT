import unittest
from registry import Registry

class TestRegistry(unittest.TestCase):
    def test_add_category(self):
        registry = Registry()
        cat_name = 'category_1'
        registry.add(cat_name)
        self.assertEqual(cat_name, registry.categories[cat_name].name)
        print(registry)

    def test_add_class(self):
        class myclass():
            def __call__(self):
                pass

        registry = Registry()
        cat_name = 'category_1'
        registry.add(cat_name, myclass)
        self.assertEqual(cat_name, registry.categories[cat_name].name)
        print(registry)

    def test_build_from_config(self):
        class myclass():
            def __init__(self, a, b):
                self.a = a
                self.b = b
            def __call__(self):
                pass

        registry = Registry()
        cat_name = 'category_1'
        registry.add(cat_name, myclass)

        config = {
            'name': 'myclass',
            'params': {'a': 1, 'b':'string'}
        }
        myobj = registry.build_from_config(cat_name, config)

        self.assertEqual(myobj.a, 1)
        self.assertEqual(myobj.b, config['params']['b'])

        print(registry)

if __name__ == '__main__':
    unittest.main()