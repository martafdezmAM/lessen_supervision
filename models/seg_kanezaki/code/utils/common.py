import torch

from seg_kanezaki.code.archs.seg_kanezaki_cnn import MyNet


def get_optimizer(args, model):
	"""
	Read from config.py file the specified optimizer and learning rate
	Args:
		args: Dictionary with parameters specified at config.py
		model: Torch DL model to be optimized

	Returns: torch.optim

	"""
	if args.OPTIMIZER == "sgd":
		optimizer = torch.optim.SGD(model.parameters(), lr=args.COMMON.LR, momentum=0.9,
		                            dampening=0, weight_decay=0.02, nesterov=False)
	elif args.OPTIMIZER == "adam":
		optimizer = torch.optim.Adam(model.parameters(), lr=args.COMMON.LR,
		                             betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
	elif args.OPTIMIZER == "asgd":
		optimizer = torch.optim.ASGD(model.parameters(), lr=args.COMMON.LR,
		                             lambd=0.0001, alpha=0.75, t0=1000000.0, weight_decay=0)
	else:
		raise ValueError("Not supported optimizer. Revise config file: OPTIMIZER")

	return optimizer


def get_model(args, input_dim):
	"""
	Create the model architecture and load the weigths
	Args:
		args: Dictionary with parameters specified at config.py
		input_dim: Input shape of the custom model

	Returns:
		torch.nn.Module: CNN model
	"""
	model = MyNet(input_dim, args)
	if args.CKPT != '':
		model.load_state_dict(torch.load(args.CKPT))

	model.cuda()
	model.train()

	return model


def get_optuna_suggest_value(trial, param_name, suggest_type, values):
    """
    Given the type of suggest and the input values return the suggested parameter.
    :param trial: Trial optuna object necessary for returning the suggested value.
    :param param_name: String identifier given as a description of the parameter.
    :param suggest_type: Reference to the optuna suggest function to be applied.
    :param values: List containing the input parameters for the suggest functions.
    :return: Value obtained by optuna trial object.
    """
    if suggest_type == "categorical":
        # Suggest a value for the categorical parameter
        return trial.suggest_categorical(name=param_name, choices=values)
    if suggest_type == "uniform":
        # Suggest a value for the continuous parameter .
        # The value is sampled from the range [low,high) in the linear domain.
        return trial.suggest_uniform(param_name, low=values[0], high=values[1])
    if suggest_type == "loguniform":
        # Suggest a value for the continuous parameter.
        # The value is sampled from the range [low,high) in the log domain.
        return trial.suggest_loguniform(param_name, low=values[0], high=values[1])
    if suggest_type == "discrete_uniform":
        # Suggest a value for the discrete parameter.
        # The value is sampled from the range [low,high], and the step of discretization is q.
        return trial.suggest_discrete_uniform(name=param_name, low=values[0], high=values[1], q=values[2])
    if suggest_type == "int":
        # Suggest a value for the integer parameter.
        # The value is sampled from the integers in [low,high], and the step of discretization is step.
        return trial.suggest_int(name=param_name, low=values[0], high=values[1], step=values[2])
    if suggest_type == "float":
        # Suggest a value for the float parameter.
        # The value is sampled from the floats in [low,high], and the step of discretization is step.
        return trial.suggest_float(name=param_name, low=values[0], high=values[1], step=values[2])

    raise ValueError  # If suggest_type is not implemented raise error
