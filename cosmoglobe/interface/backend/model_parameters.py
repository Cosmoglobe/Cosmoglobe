from __future__ import annotations

from pydantic import BaseModel
from enum import Enum, auto
from typing import Callable
from functools import partial

from cg_sampling_group import CGSamplingGroup
from component import Component

class ModelParameters(BaseModel):
    """
    A container for the general Commander parameters that define the models used in
    Commander. Also contains a list of Component containers, which contain
    model-specific parameters, and a list of CGSamplingGroup containers, each
    of which defines a CG sampling group.
    """
    instrument_param_file: str = None
    init_instrument_from_hdf: str = None
    cg_sampling_groups: list[CGSamplingGroup] = None
    signal_components: list[Component] = None

    @classmethod
    def _get_parameter_handling_dict(cls) -> dict[str, Callable]:
        """
        Create a mapping between the container field names and the appropriate
        functions to handle those fields.

        The functions in the output dict will take a parameter file dictionary
        as the only argument (with the exception of the handling of
        the 'cg_sampling_groups' field, which also will take an instantiated
        list of Component instances), and will return whatever is appropriate
        for that field.

        Output:
            dict[str, Callable]: Mapping between field names and their handlers.
        """

        def default_handler(field_name, paramfile_dict):
            paramfile_param = field_name.upper()
            try:
                return paramfile_dict[paramfile_param]
            except KeyError as e:
                print("Warning: Model parameter {} not found in parameter file".format(e))
                return None

        def signal_component_handler(field_name, paramfile_dict):
            signal_components = []
            num_components = int(paramfile_dict['NUM_SIGNAL_COMPONENTS'])
            for i in range(1, num_components+1):
                signal_components.append(
                    Component.create_component_params(paramfile_dict, i))
            return signal_components

        def cg_sampling_group_handler(field_name, paramfile_dict, signal_components):
            cg_sampling_groups = []
            num_sampling_groups = int(paramfile_dict['NUM_CG_SAMPLING_GROUPS'])
            for i in range(1, num_sampling_groups + 1):
                cg_sampling_groups.append(
                    CGSamplingGroup.create_cg_sampling_group(
                        paramfile_dict, i, signal_components))
            return cg_sampling_groups

        field_names = cls.__fields__.keys()
        handling_dict = {}
        for field_name in field_names:
            if field_name == 'cg_sampling_groups':
                handling_dict[field_name] = partial(
                    cg_sampling_group_handler, field_name)
            elif field_name == 'signal_components':
                handling_dict[field_name] = partial(
                    signal_component_handler, field_name)
            else:
                handling_dict[field_name] = partial(
                    default_handler, field_name)
        return handling_dict

    @classmethod
    def create_model_params(cls,
                            paramfile_dict: dict[str, Any]) -> ModelParameters:
        """
        Factory class method for a ModelParameters instance.

        Input:
            paramfile_dict[str, str]: A dict (typically created by
                parameter_parser._paramfile_to_dict) mapping the keys found in
                a Commander parameter file to the values found in that same
                file.
        Output:
            ModelParameters: Parameter container for the model-specific
                Commander parameters. It will also point to a list of Component
                parameter collection instances, as well as a list of
                CGSamplingGroup instances.
        """

        handling_dict = cls._get_parameter_handling_dict()
        param_vals = {}
        for field_name, handling_function in handling_dict.items():
            if field_name == 'cg_sampling_groups':
                # This must be done *after* the components have been created and
                # populated in order to point to them.
                continue
            param_vals[field_name] = handling_function(paramfile_dict)
        param_vals['cg_sampling_groups'] = handling_dict['cg_sampling_groups'](
            paramfile_dict, param_vals['signal_components'])
        return ModelParameters(**param_vals)

    def serialize_to_paramfile_dict(self):
        """
        Creates a mapping from Commander parameter names to the values in the
        ModelParameters instance, with all lower-level parameter collections
        similarly serialized.

        Note the values in this mapping are basic types, not strings. This
        means they will have to be processed further before they are ready for
        a Commander parameter file. The keys, however, need no more processing.

        Output:
            dict[str, Any]: Mapping from Commander parameter file names to the
                parameter values.
        """
        paramfile_dict = {}
        for field_name, value in self.__dict__.items():
            if field_name == 'cg_sampling_groups':
                num_cg_sampling_groups = len(value)
                paramfile_dict['NUM_CG_SAMPLING_GROUPS'] = num_cg_sampling_groups
                for i, cg_sampling_group in enumerate(value):
                    paramfile_dict.update(
                        cg_sampling_group.serialize_to_paramfile_dict(i+1))
            elif field_name == 'signal_components':
                num_signal_components = len(value)
                paramfile_dict['NUM_SIGNAL_COMPONENTS'] = num_signal_components
                for i, component in enumerate(value):
                    paramfile_dict.update(
                        component.serialize_to_paramfile_dict(i+1))
            else:
                paramfile_dict[field_name.upper()] = value
        return paramfile_dict
