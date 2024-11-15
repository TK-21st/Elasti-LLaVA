from contextlib import contextmanager

# Global variable to track the flex hooks state
# always true unless disabled
_ELASTIC_MODE = True

@contextmanager
def disable_elastic_mode():
    global _ELASTIC_MODE
    previous_state = _ELASTIC_MODE
    _ELASTIC_MODE = False
    try:
        yield
    finally:
        _ELASTIC_MODE = previous_state

def is_elastic_mode():
    global _ELASTIC_MODE
    return _ELASTIC_MODE
