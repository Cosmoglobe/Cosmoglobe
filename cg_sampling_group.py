from enum import Enum, auto
from pydantic import BaseModel
from functools import partial

from component import Component

class CGSamplingGroup(BaseModel):
    mask: str
    maxiter: int
    components: list[Component]

    @classmethod
    def _get_parameter_handling_dict(cls):

        def default_handler(field_name, paramfile_dict, cg_sampling_group_num,
                            components):
            paramfile_param = (
                'CG_SAMPLING_GROUP_' +
                field_name.upper() + 
                '{:02d}'.format(cg_sampling_group_num))
            try:
                return paramfile_dict[paramfile_param]
            except KeyError as e:
                print("Warning: CG sampling group parameter {} not found in parameter file".format(e))
                return None


        def component_handler(field_name,
                              paramfile_dict,
                              cg_sampling_group_num,
                              components):
            compnames = paramfile_dict['CG_SAMPLING_GROUP{:02d}'.format(
                cg_sampling_group_num)]
            compnames = compnames.replace("'", "")
            compnames = compnames.split(',')
            linked_components = []
            for compname in compnames:
                for component in components:
                    if component.label == compname:
                        linked_components.append(component)
                        break
                else:
                    raise ValueError("Component {} specified in CG sampling group but is not defined in parameter file")
            return linked_components
        field_names = cls.__fields__.keys()
        handling_dict = {}
        for field_name in field_names:
            if field_name == 'components':
                handling_dict[field_name] = partial(component_handler, field_name)
            else:
                handling_dict[field_name] = partial(default_handler, field_name)
        return handling_dict

    @classmethod
    def create_cg_sampling_group(cls, paramfile_dict, cg_sampling_group_num,
                                 components):
        handling_dict = cls._get_parameter_handling_dict()
        param_vals = {}
        for field_name, handling_function in handling_dict.items():
            param_vals[field_name] = handling_function(paramfile_dict,
                                                       cg_sampling_group_num,
                                                       components)
        return CGSamplingGroup(**param_vals)
