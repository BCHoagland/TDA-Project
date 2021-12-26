# wraps dict so you can access a key's value by dict.KEY instead of dict['KEY']

class Hyperparams(dict):

    def __getattr__(self, item):
        return super().__getitem__(item)

    def __setattr__(self, item, value):
        return super().__setitem__(item, value)
