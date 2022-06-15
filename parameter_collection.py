from dataclasses import dataclass

@dataclass
class ParameterCollection:

    def set_param(self, param_name, param_value):
        setattr(self, param_name, param_value)
