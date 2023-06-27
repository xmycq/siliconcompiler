import logging
import uuid
import math
import os
import multiprocessing

try:
    from vizier.service import clients as vz_clients
    from vizier.service import pyvizier as vz

    from jax import config
    config.update("jax_enable_x64", True)
    __has_vizier = True

    logging.getLogger('absl').setLevel(logging.CRITICAL)
    logging.getLogger('jax').setLevel(logging.CRITICAL)
except ModuleNotFoundError:
    __has_vizier = False


def __make_parameter_entry(key, step, index):
    param_name = ','.join(key)

    key_str = f'{key}'
    node_name = None
    if step is not None:
        param_name += f'-{step}'
        node_name = step

        if index is not None:
            param_name += f'-{index}'
            node_name += f'{index}'

    if node_name:
        key_str += f' ({node_name})'

    return param_name, {
        "key": key,
        "step": step,
        "index": index,
        "print": key_str
    }


def __add_vizier_parameters(chip, search_space, parameters):
    parameter_map = {}

    for param in parameters:
        key = param['key']
        values = param['values']
        if 'type' in param:
            key_type = param['type']
        else:
            key_type = chip.get(*key, field='type')
            if key_type.startswith('['):
                key_type = key_type[1:-1]

        step = None
        index = None
        if 'step' in param:
            step = param['step']
        if 'index' in param:
            index = param['index']

        param_name, param_entry = __make_parameter_entry(key, step, index)
        parameter_map[param_name] = param_entry

        if key_type == 'float':
            search_space.root.add_float_param(param_name, values[0], values[1])
        elif key_type == 'int':
            search_space.root.add_int_param(param_name, values[0], values[1])
        elif key_type == 'discrete':
            search_space.root.add_discrete_param(param_name, values)
        elif key_type == 'bool':
            if not values:
                values = ['true', 'false']
            search_space.root.add_discrete_param(param_name, values)
        elif key_type == 'enum':
            search_space.root.add_categorical_param(param_name, values)
        else:
            raise ValueError(f'{key_type} is not supported')

    return parameter_map


def __add_vizier_measurement(metric_information, goals):
    measurement_map = {}
    for goal in goals:
        key = goal['key']
        target = goal['target']

        step = None
        index = None
        if 'step' in goal:
            step = goal['step']
        if 'index' in goal:
            index = goal['index']

        goal_name, goal_entry = __make_parameter_entry(key, step, index)
        measurement_map[goal_name] = goal_entry

        vz_goal = None
        if target == 'max':
            vz_goal = vz.ObjectiveMetricGoal.MAXIMIZE
        elif target == 'min':
            vz_goal = vz.ObjectiveMetricGoal.MINIMIZE
        else:
            raise ValueError(f'{target} is not a supported goal')

        metric_information.append(vz.MetricInformation(goal_name, goal=vz_goal))

    return measurement_map


def _optimize_vizier(chip, parameters, goals, experiments, parallel_limit=None):
    if not __has_vizier:
        chip.logger.error('Vizier is not available')
        return

    if not experiments:
        experiments = 10 * len(parameters)
        chip.logger.debug(f'Setting number of optimizer experiments to {experiments}')

    # Algorithm, search space, and metrics.
    study_config = vz.StudyConfig()
    parameter_map = __add_vizier_parameters(chip, study_config.search_space, parameters)
    measurement_map = __add_vizier_measurement(study_config.metric_information, goals)

    # Setup client and begin optimization. Vizier Service will be implicitly created.
    study = vz_clients.Study.from_study_config(study_config,
                                               owner=chip.design,
                                               study_id=uuid.uuid4().hex)

    if not parallel_limit:
        parallel_limit = 1

    if not chip.get('option', 'remote'):
        parallel_limit = 1

    rounds = int(math.ceil(float(experiments) / parallel_limit))

    multiprocessor = multiprocessing.get_context('spawn')

    for n in range(rounds):
        trial_chips = {}

        # Setup each experiment
        for m, suggestion in enumerate(study.suggest(count=parallel_limit)):
            trial_chip = chip._copy()
            trial_chips[m] = {
                "chip": trial_chip,
                "failed": None,
                "run": multiprocessor.Process(target=trial_chip.run),
                "suggestion": suggestion
            }

            jobname = f"{chip.get('option', 'jobname')}_optimize_{n+1}_{m+1}"

            chip.logger.info(f'Setting parameters for {jobname}')
            for param_name, param_value in suggestion.parameters.items():
                param_entry = parameter_map[param_name]
                chip.logger.info(f'Setting {param_entry["print"]} = {param_value}')
                trial_chip.set(*param_entry["key"], str(param_value),
                               step=param_entry["step"], index=param_entry["index"])

            trial_chip.set('option', 'jobname', jobname)
            trial_chip.set('option', 'quiet', True)

        # Start run
        for trial_entry in trial_chips.values():
            trial_chip = trial_entry['chip']
            trial_runner = trial_entry['run']
            chip.logger.info(f"Starting optimizer run ({trial_chip.get('option', 'jobname')})")
            trial_runner.start()

        # Wait for them to finish
        for trial_entry in trial_chips.values():
            trial_entry['run'].join()

        # Record results
        for trial_entry in trial_chips.values():
            trial_chip = trial_entry['chip']
            trial_suggestion = trial_entry['suggestion']
            # Read back the final manifest
            trial_chip.read_manifest(
                os.path.join(trial_chip._getworkdir(), f'{trial_chip.design}.pkg.json'))

            measurement = {}
            for meas_name, meas_entry in measurement_map.items():
                measurement[meas_name] = trial_chip.get(
                    *meas_entry["key"],
                    step=meas_entry["step"], index=meas_entry["index"])
                trial_chip.logger.info(f'Measured {meas_entry["print"]} = {measurement[meas_name]}')

            if any([value is None for value in measurement.values()]):
                trial_entry['failed'] = "Did not record measurement goal"

            if trial_entry['failed']:
                chip.logger.error(f'{jobname} failed: {trial_entry["failed"]}')
                trial_suggestion.complete(vz.Measurement(),
                                          infeasible_reason=trial_entry['failed'])
            else:
                trial_suggestion.complete(vz.Measurement(measurement))
            chip.schema.cfg['history'][jobname] = trial_chip.schema.history(jobname).cfg

    optimal_trials = list(study.optimal_trials())
    for n, optimal_trial in enumerate(optimal_trials):
        optimal_trial = optimal_trial.materialize()

        chip.logger.info(f"Optimal settings {n+1}/{len(optimal_trials)}:")
        chip.logger.info("Parameters:")
        trial_parameters = optimal_trial.parameters
        for param_name, param_key in trial_parameters.items():
            param_print = parameter_map[param_name]['print']
            chip.logger.info(f"  {param_print} = {param_key}")

        chip.logger.info("Measurements:")
        trial_objectives = optimal_trial.final_measurement
        for meas_name, meas_key in trial_objectives.metrics.items():
            goal_print = measurement_map[meas_name]['print']
            chip.logger.info(f"  {goal_print} = {meas_key.value}")

    # Remove study from DB
    study.delete()
