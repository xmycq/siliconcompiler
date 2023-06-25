import logging
import uuid

try:
    from vizier.service import clients as vz_clients
    from vizier.service import pyvizier as vz

    from jax import config
    config.update("jax_enable_x64", True)
    __has_vizier = True
except ModuleNotFoundError:
    __has_vizier = False


def _optimize_vizier(chip, parameters, goals, rounds):
    if not __has_vizier:
        chip.logger.error('Vizier is not available')
        return

    parameter_map = {}

    logging.getLogger('absl').setLevel(logging.CRITICAL)
    logging.getLogger('jax').setLevel(logging.CRITICAL)

    # Algorithm, search space, and metrics.
    study_config = vz.StudyConfig()
    for param in parameters:
        key = param['key']
        values = param['values']
        if 'type' in param:
            key_type = param['type']
        else:
            key_type = chip.get(*key, field='type')
            if key_type.startswith('['):
                key_type = key_type[1:-1]

        param_name = ','.join(key)
        parameter_map[param_name] = key

        if key_type == 'float':
            study_config.search_space.root.add_float_param(param_name,
                                                           values[0], values[1])
        elif key_type == 'int':
            study_config.search_space.root.add_int_param(param_name,
                                                         values[0], values[1])
        elif key_type == 'discrete':
            study_config.search_space.root.add_discrete_param(param_name,
                                                              values)
        elif key_type == 'bool':
            study_config.search_space.root.add_discrete_param(param_name,
                                                              ['true', 'false'])
        elif key_type == 'enum':
            study_config.search_space.root.add_categorical_param(param_name,
                                                                 values)
        else:
            raise ValueError(f'{key_type} is not supported')

    measurement_map = {}
    for goal in goals:
        key = goal['key']
        target = goal['target']

        goal_name = ','.join(key)
        measurement_map[goal_name] = key

        vz_goal = None
        if target == 'max':
            vz_goal = vz.ObjectiveMetricGoal.MAXIMIZE
        elif target == 'min':
            vz_goal = vz.ObjectiveMetricGoal.MINIMIZE
        else:
            raise ValueError(f'{target} is not a supported goal')

        study_config.metric_information.append(vz.MetricInformation(goal_name, goal=vz_goal))

    # Setup client and begin optimization. Vizier Service will be implicitly created.
    study = vz_clients.Study.from_study_config(study_config,
                                               owner=chip.design,
                                               study_id=uuid.uuid4().hex)

    for n in range(rounds):
        for suggestion in study.suggest(count=1):
            trial_chip = chip._copy()

            for param_name, param_value in suggestion.parameters.items():
                key = parameter_map[param_name]
                trial_chip.logger.info(f'Setting {key} = {param_value}')
                trial_chip.set(*key, str(param_value), step='place', index='0')

            jobname = f"{chip.get('option', 'jobname')}_optimize_{n}"

            trial_chip.logger.info(f'Starting optimizer run {n}')
            trial_chip.set('option', 'jobname', jobname)
            trial_chip.set('option', 'quiet', True)
            trial_chip.set('option', 'resume', True)

            try:
                trial_chip.run()
            except Exception as e:
                suggestion.complete(infeasible_reason=f"{e}")
                continue

            measurement = {}
            for meas_name, meas_key in measurement_map.items():
                measurement[meas_name] = chip.get(*meas_key, step='export', index='1')
                trial_chip.logger.info(f'Measured {meas_key} = {measurement[meas_name]}')

            if any([value is None for value in measurement.values()]):
                suggestion.complete(infeasible_reason="Did not record measurement goal")
            else:
                suggestion.complete(vz.Measurement(measurement))

            chip.schema.cfg['history'][jobname] = trial_chip.schema.history(jobname).cfg

    for n, optimal_trial in enumerate(study.optimal_trials()):
        optimal_trial = optimal_trial.materialize()

        chip.logger.info(f"Optimal Trial Suggestion and Objective {n}:")
        chip.logger.info("Suggestion:")
        trial_parameters = optimal_trial.parameters
        for param_name, param_key in trial_parameters.items():
            chip.logger.info(f"  {param_name} = {param_key}")
        chip.logger.info("Objective:")
        trial_objectives = optimal_trial.final_measurement
        for meas_name, meas_key in trial_objectives.metrics.items():
            chip.logger.info(f"  {meas_name} = {meas_key.value}")

    study.delete()
