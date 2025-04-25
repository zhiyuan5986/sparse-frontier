class GlobalSettings:
    _settings = {}

    @classmethod
    def get(cls, key, default=None):
        return cls._settings.get(key, default)

    @classmethod
    def set(cls, key, value):
        cls._settings[key] = value


def is_vllm_profiling_done():
    return GlobalSettings.get("vllm_profiling_done", False)


def set_vllm_profiling_done(done: bool):
    GlobalSettings.set("vllm_profiling_done", done)
