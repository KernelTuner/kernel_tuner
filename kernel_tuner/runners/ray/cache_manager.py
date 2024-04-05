import ray
import json

from kernel_tuner.util import store_cache

@ray.remote
class CacheManager:
    def __init__(self, tuning_options):
        self.tuning_options = tuning_options

    def store(self, key, params):
        store_cache(key, params, self.tuning_options)

    def check_and_retrieve(self, key):
        """Checks if a result exists for the given key and returns it if found."""
        if self.tuning_options.cache:
            return self.tuning_options.cache.get(key, None)
        else:
            return None
    
    def get_tuning_options(self):
        """Returns the current tuning options."""
        return self.tuning_options
