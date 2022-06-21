from pydantic import BaseModel, root_validator

class ParameterCollection(BaseModel):

    @root_validator(pre=True)
    def strip_fortran_bool(cls, params):
        curated_vals = {}
        for key, value in params.items():
            if isinstance(value, str) and value.startswith('.') and value.endswith('.'):
                curated_vals[key] = value[1:-1]
            else:
                curated_vals[key] = value
        return curated_vals

    @root_validator(pre=True)
    def defortranify_floats(cls, params):
        curated_vals = {}
        for key, value in params.items():
            if isinstance(value, str) and len(value) > 2 and 'd' in value.lower() and (value[0].isdigit() or value[1].isdigit()):
                curated_vals[key] = value.lower().replace('d', 'e')
#            if isinstance(value, str) and value.startswith('.') and value.endswith('.'):
#                curated_vals[key] = value[1:-1]
            else:
                curated_vals[key] = value
        return curated_vals


    def set_param(self, param_name, param_value):
        setattr(self, param_name, param_value)
