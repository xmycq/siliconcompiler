
def run_node(chip, step, index, start_time=0.0):
    '''
    "Local run" dispatcher.

    This plugin runs a flowgraph node on the local machine.
    This is the default behavior if no other dispatch method is specified.
    '''

    chip._executenode(step, index)
    chip._finalizenode(step, index, start_time)
