import os

__all__ = [
    "FloatInRange",
    "IntInRange",
    "MultipleOf",
    "Bool",
    "Percentage",
    "OddInt",
    "EvenInt",
    "PositiveInt",
    "PositiveFloat",
    "FilePath",
    "DirPath",
]


## Types with specific behavior
class FloatInRange:
    """Positive float class for config parsing"""

    def __init__(self, min_value, max_value):
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, value):
        if value < self.min_value or value > self.max_value:
            raise ValueError(
                "Float must be between {} and {}".format(self.min_value, self.max_value)
            )
        return float.__new__(value)


class IntInRange:
    """Integer in range class for config parsing"""

    def __init__(self, min_value, max_value):
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, value):
        if value < self.min_value or value > self.max_value:
            raise ValueError(
                "Integer must be between {} and {}".format(
                    self.min_value, self.max_value
                )
            )
        return int.__new__(value)


class MultipleOf:
    """MultipleOf class for config parsing"""

    def __init__(self, multiple):
        self.multiple = multiple

    def __call__(self, value):
        if value % self.multiple != 0:
            raise ValueError("Integer must be a multiple of {}".format(self.multiple))
        return int(value)


## Types overloading built-in python types


class Bool:
    """Boolean class for config parsing"""

    def __new__(cls, value):
        if str(value).lower() in ["true", "1", "t", "y", "yes"]:
            return True
        elif str(value).lower() in ["false", "0", "f", "n", "no"]:
            return False
        else:
            raise ValueError(
                "Boolean value must be one of true, false, 1, 0, t, f, y, n, yes, no"
            )


class OddInt(int):
    """Odd integer class for config parsing"""

    def __new__(cls, value):
        if value % 2 == 0:
            raise ValueError("Odd integer must be odd")
        return int.__new__(cls, value)


class EvenInt(int):
    """Even integer class for config parsing"""

    def __new__(cls, value):
        if value % 2 != 0:
            raise ValueError("Even integer must be even")
        return int.__new__(cls, value)


class Percentage(float):
    """Percentage class for config parsing"""

    def __new__(cls, value):
        if value < 0 or value > 1:
            raise ValueError("Percentage must be between 0 and 1")
        return float.__new__(cls, value)


class PositiveInt(int):
    """Positive integer class for config parsing"""

    def __new__(cls, value):
        if value < 0:
            raise ValueError("Positive integer must be greater than 0")
        return int.__new__(cls, value)


class PositiveFloat(float):
    """Positive float class for config parsing"""

    def __new__(cls, value):
        if value < 0:
            raise ValueError("Positive float must be greater than 0")
        return float.__new__(cls, value)


## Types for file and directory paths


class FilePath(str):
    """File path class for config parsing"""

    def __new__(cls, value):
        if not os.path.exists(value):
            raise ValueError("File path does not exist")
        return str.__new__(cls, value)


class DirPath(str):
    """Directory path class for config parsing"""

    def __new__(cls, value):
        if not os.path.isdir(value):
            raise ValueError("Directory path does not exist")
        return str.__new__(cls, value)
