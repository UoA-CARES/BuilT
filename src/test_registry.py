import unittest
from registry import registry as r

class TestRegistry(unittest.TestCase):
    def test_add_category(self):
        # registry = Registry()
        r.clear()

        cat_name = 'category_1'
        r.add(cat_name)
        self.assertEqual(cat_name, r.categories[cat_name].name)
        print(r)

    def test_add_class(self):
        class myclass():
            def __call__(self):
                pass

        class myclass2():
            def __call__(self):
                pass

        r.clear() 

        cat_name = 'category_1'
        r.add(cat_name, myclass)
        self.assertEqual(cat_name, r.categories[cat_name].name)        

        r.add(cat_name, myclass2)

        self.assertEqual(len(r.categories), 1) 
        self.assertEqual(len(r.categories[cat_name].class_dict), 2)
        self.assertIn(myclass.__name__, r.categories[cat_name].class_dict.keys())
        self.assertIn(myclass2.__name__, r.categories[cat_name].class_dict.keys())

    def test_build_from_config(self):
        class myclass():
            def __init__(self, a, b):
                self.a = a
                self.b = b
            def __call__(self):
                pass

        r.clear()

        cat_name = 'category_1'
        r.add(cat_name, myclass)

        config = {
            'name': 'myclass',
            'params': {'a': 1, 'b':'string'}
        }
        myobj = r.build_from_config(cat_name, config)

        self.assertEqual(myobj.a, 1)
        self.assertEqual(myobj.b, config['params']['b'])

        print(r)

if __name__ == '__main__':
    unittest.main()