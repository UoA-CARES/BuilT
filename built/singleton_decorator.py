class SingletonDecorator:
    """A decorator for the Singleton pattern"""
    def __init__(self, clazz):
        self.clazz = clazz
        self.instance = None
    def __call__(self, *args, **kwds):
        if self.instance == None:
            self.instance = self.clazz(*args, **kwds)
        return self.instance