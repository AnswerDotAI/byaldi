from io import StringIO
import sys

def capture_print(func):
    def wrapper(*args, **kwargs):
        original_stdout = sys.stdout
        sys.stdout = StringIO()
        try:
            result = func(*args, **kwargs)
        finally:
            sys.stdout = original_stdout
        return result
    return wrapper