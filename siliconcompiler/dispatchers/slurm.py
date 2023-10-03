
from siliconcompiler import scheduler

def run_node(chip, step, index, start_time=0.0):
    '''
    Slurm dispatcher.

    This plugin runs a flowgraph node on a Slurm cluster which the
    local machine has access to.

    The SC command will be submitted to the scheduler using an "sbatch"
    call, with options configured in the ('option', 'scheduler') Schema
    keypaths.

    If the Slurm cluster requires setup/teardown steps on its compute nodes,
    the default run script can be overridden by writing a shell script in:
        [job build directory]/configs/[step][index].sh
    This script should contain an `sc` command or build script which runs
    the given step using the ('arg', 'step') and ('arg', 'index') parameters.
    '''

    # TODO: Move this logic out of 'scheduler.py' and delete the old file.
    scheduler._defernode(chip, step, index)
