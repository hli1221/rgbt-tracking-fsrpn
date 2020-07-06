import random


class TrackerParams:
    """Class for tracker parameters."""
    def free_memory(self):
        for a in dir(self):
            if not a.startswith('__') and hasattr(getattr(self, a), 'free_memory'):
                getattr(self, a).free_memory()


