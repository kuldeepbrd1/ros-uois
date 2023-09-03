class Config:
    def _check_types(dict_to_check: dict, type_dict: dict):
        """Check if all keys in dict_to_check are in type_dict and have the correct type

        Args:
            dict_to_check (dict): dictionary to check
            type_dict (dict): dictionary containing the types of the keys in dict_to_check

        Returns:
            bool: True if all keys in dict_to_check are in type_dict and have the correct type
        """
        for k, v in type_dict.items():
            if k not in dict_to_check:
                raise KeyError("Config key {} not found".format(k))

            if isinstance(v, dict):
                Config._check_types(dict_to_check[k], v)
                continue

            try:
                dict_to_check[k] = v(dict_to_check[k])
            except ValueError as e:
                raise ValueError("Config key {} is invalid: {}".format(k, e))

        return True
