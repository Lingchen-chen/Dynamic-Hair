import importlib
from .base_solver import *


def find_model_using_name(model_name):
    """Import "Models/[model_name]Solver.py"."""
    model_name += "Solver"
    model_file_name = "Models." + model_name
    modellib = importlib.import_module(model_file_name)
    model = None
    for name, cls in modellib.__dict__.items():
        if name.lower() == model_name.lower() and issubclass(cls, BaseSolver):
            model = cls

    if model is None:
        print("In %s.py, there should be a subclass of BaseSolver with class name that matches %s in lowercase." % (model_file_name, model_name))
        exit(0)

    return model


def get_option_setter(model_name):
    """Return the static method <modify_commandline_options> of the model class."""
    model = find_model_using_name(model_name)
    return model.modify_options


def get_option_parser(model_name):
    """Return the static method <parse_commandline_options> of the model class."""
    model = find_model_using_name(model_name)
    return model.parse_opt