from __future__ import annotations

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
        """
        Create a mapping between the container field names and the appropriate
        functions to handle those fields.

        The functions in the output dict will take a parameter file dictionary,
        the cg sampling group number, and a list of instantiated Component
        instances as arguments, and will return whatever is appropriate for
        that field.

        Output:
            dict[str, Callable]: Mapping between field names and their handlers.
        """

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
    def create_cg_sampling_group(cls,
                                 paramfile_dict: dict[str, Any],
                                 cg_sampling_group_num: int,
                                 components: list[Component]) -> CGSamplingGroup:
        """
        Factory class method for a CGSamplingGroup instance.

        Input:
            paramfile_dict[str, str]: A dict (typically created by
                parameter_parser._paramfile_to_dict) mapping the keys found in
                a Commander parameter file to the values found in that same
                file.
            cg_sampling_group_num (int): The number of the CG sampling group to
                be instantiated.
            components (list[Component]): The Components to which the parent
                ModelParameters instance of the CG sampling group is pointing,
                in order to link a CG sampling group with the right components.
        Output:
            CGSamplingGroup: Parameter container for a CG sampling
                group-specific set of Commander parameters.
        """

        handling_dict = cls._get_parameter_handling_dict()
        param_vals = {}
        for field_name, handling_function in handling_dict.items():
            param_vals[field_name] = handling_function(paramfile_dict,
                                                       cg_sampling_group_num,
                                                       components)
        return CGSamplingGroup(**param_vals)

    def serialize_to_paramfile_dict(self, cg_sampling_group_num):
        """
        Creates a mapping from Commander parameter names to the values in the
        CGSamplingGroup instance, with all lower-level parameter collections
        similarly serialized.

        Note the values in this mapping are basic types, not strings. This
        means they will have to be processed further before they are ready for
        a Commander parameter file. The keys, however, need no more processing.

        Input:
            cg_sampling_group_num[int]: The number of the cg sampling group
            instance in the Commander file context.

        Output:
            dict[str, Any]: Mapping from Commander parameter file names to the
                parameter values.
        """

        paramfile_dict = {}
        for field_name, value in self.__dict__.items():
            if field_name == 'components':
                comp_name_list = []
                for component in value:
                    comp_name_list.append(component.label)
                paramfile_dict[
                    'CG_SAMPLING_GROUP{:02}'.format(cg_sampling_group_num)] = (
                        ','.join(comp_name_list))
            else:
                paramfile_dict[
                    'CG_SAMPLING_GROUP_{}{:02}'.format(
                        field_name.upper(), cg_sampling_group_num)] = value
        return paramfile_dict
