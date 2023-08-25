import numpy as np


# Dict which accepts np bool arrays as keys
class NumpyDict:
    def __init__(self):
        self.core_dict = {}

    def __setitem__(self, key, value):
        parsed_key = np.packbits(key).tobytes()
        self.core_dict[parsed_key] = value

    def __getitem__(self, item: np.ndarray):
        parsed_key = np.packbits(item).tobytes()
        try:
            return self.core_dict[parsed_key]
        except KeyError:
            to_print = ""
            for row in item:
                for col in row:
                    to_print += "█" if col else " "
                to_print += "\n"
            print(to_print)
            raise
