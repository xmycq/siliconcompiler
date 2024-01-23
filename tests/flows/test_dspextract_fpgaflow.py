import os

import pytest

import siliconcompiler

from siliconcompiler.targets import fpgaflow_demo


@pytest.mark.eda
@pytest.mark.quick
def test_dspextract_fpgaflow(scroot,
                             arch_name='example_arch_X030Y030'):

    top_module = 'macc_pipe'

    chip = siliconcompiler.Chip(f'{top_module}')

    chip.set('fpga', 'partname', arch_name)

    # This example architecture doesn't have a provided routing
    # graph file, so we don't have the metadata to to bitstream
    # generation.  Stop after routing instead of running to
    # completion.
    chip.set('option', 'to', ['route'])

    flow_root = os.path.join(scroot, 'examples', 'fpga_flow')

    # 1. Defining the project
    # 2. Define source files
    v_src = os.path.join(flow_root, 'designs', top_module, f'{top_module}.v')
    chip.input(v_src)

    # 3. Load target
    chip.load_target(fpgaflow_demo)

    assert chip.check_filepaths()

    chip.run()

    route_file = chip.find_result('route', step='route', index='0')
    assert route_file
    assert os.path.exists(route_file)


if __name__ == "__main__":
    scroot = os.path.abspath(__file__).replace("/tests/flows/test_dspextract_fpgaflow.py", "")
    test_dspextract_fpgaflow(scroot)
