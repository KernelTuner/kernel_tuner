"""Ray caching module."""
import ray

from kernel_tuner.util import store_cache


@ray.remote(num_cpus=1)
class CacheManager(object):
    """Manage the cache used by the parallel actors."""

    def __init__(self, cache, cachefile):
        # importing here due to circular import
        from kernel_tuner.interface import Options

        self.tuning_options = Options({"cache": cache, "cachefile": cachefile})

    def store(self, key, params):
        store_cache(key, params, self.tuning_options)

    def check_and_retrieve(self, key):
        """Checks if a result exists for the given key and returns it if found."""
        if self.tuning_options["cache"]:
            return self.tuning_options["cache"].get(key, None)
        else:
            return None

    def get_cache(self):
        """Returns the current tuning options."""
        return self.tuning_options["cache"], self.tuning_options["cachefile"]
