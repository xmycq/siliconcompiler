'''
VPR (Versatile Place and Route) is an open source CAD
tool designed for the exploration of new FPGA architectures and
CAD algorithms, at the packing, placement and routing phases of
the CAD flow. VPR takes, as input, a description of an FPGA
architecture along with a technology-mapped user circuit. It
then performs packing, placement, and routing to map the
circuit onto the FPGA. The output of VPR includes the FPGA
configuration needed to implement the circuit and statistics about
the final mapped design (eg. critical path delay, area, etc).

Documentation: https://docs.verilogtorouting.org/en/latest

Sources: https://github.com/verilog-to-routing/vtr-verilog-to-routing

Installation: https://github.com/verilog-to-routing/vtr-verilog-to-routing
'''

import os
import shutil
import json
import re


__block_file = "reports/block_usage.json"


######################################################################
# Make Docs
######################################################################
def make_docs(chip):
    chip.set('fpga', 'partname', 'example_arch_X005Y005')
    chip.load_target("fpgaflow_demo")
    setup_tool(chip)
    return chip


def setup_tool(chip, clobber=True):

    chip.set('tool', 'vpr', 'exe', 'vpr', clobber=clobber)
    chip.set('tool', 'vpr', 'vswitch', '--version')
    chip.set('tool', 'vpr', 'version', '>=8.1.0', clobber=clobber)

    step = chip.get('arg', 'step')
    index = chip.get('arg', 'index')
    tool, task = chip._get_tool_task(step, index)
    chip.set('tool', tool, 'task', task, 'regex', 'warnings', "^Warning",
             step=step, index=index, clobber=False)
    chip.set('tool', tool, 'task', task, 'regex', 'errors', "^Error",
             step=step, index=index, clobber=False)


def runtime_options(chip, tool='vpr'):

    part_name = chip.get('fpga', 'partname')
    step = chip.get('arg', 'step')
    index = chip.get('arg', 'index')
    task = chip._get_task(step, index)

    design = chip.top()

    options = []

    options.append(f"--write_block_usage {__block_file}")
    options.append("--outfile_prefix outputs/")

    topmodule = chip.top()
    blif = f"inputs/{topmodule}.blif"

    if chip.valid('fpga', part_name, 'file', 'archfile') and \
       chip.get('fpga', part_name, 'file', 'archfile'):

        archs = chip.find_files('fpga', part_name, 'file', 'archfile',
                                step=None, index=None)

    else:
        archs = []

    if (len(archs) == 1):
        options.append(archs[0])
    elif (len(archs) == 0):
        chip.error("VPR requires an architecture file as one of its command line arguments",
                   fatal=True)
    else:
        chip.error("Only one architecture XML file can be passed to VPR", fatal=True)

    options.append(blif)

    threads = chip.get('tool', tool, 'task', task, 'threads', step=step, index=index)
    options.append(f"--num_workers {threads}")

    if (task == 'place'):
        # Confine VPR execution to packing and placement steps
        options.append('--pack')
        options.append('--place')
    elif (task == 'route'):
        options.append('--route')
        # To run only the routing step we need to pass in the placement files
        options.append(f'--net_file inputs/{design}.net')
        options.append(f'--place_file inputs/{design}.place')
    elif (task == 'bitstream'):
        options.append(f'--net_file inputs/{design}.net')
        options.append(f'--place_file inputs/{design}.place')
        options.append(f'--route_file inputs/{design}.route')
    else:
        chip.error(f"Specified task {task} doesn't map to a VPR operation", fatal=True)

    if 'sdc' in chip.getkeys('input', 'constraint'):
        sdc_file = find_single_file(chip, 'input', 'constraint', 'sdc',
                                    step=step, index=index,
                                    file_not_found_msg="SDC file not found")
        if (sdc_file is not None):
            sdc_arg = f"--sdc_file {sdc_file}"
            options.append(sdc_arg)
    else:
        options.append("--timing_analysis off")

    # Give input constraint file priority over individually-specified
    # constraints that our place step pre-processing may have generated
    # from chip.get('constraint', 'placement', ...) settings:
    if 'pins' in chip.getkeys('input', 'constraint'):
        pin_constraint_file = find_single_file(chip, 'input', 'constraint', 'pins',
                                               step=step, index=index,
                                               file_not_found_msg="VPR constraints file not found")
        if (pin_constraint_file is not None):
            pin_constraint_arg = f"--read_vpr_constraints {pin_constraint_file}"
            options.append(pin_constraint_arg)
    # If no constraint file, look for the place preprocessing to have
    # dumped out a file (see above for specified path/name)
    elif (os.path.isfile(auto_constraints())):
        pin_constraint_arg = f"--read_vpr_constraints {auto_constraints()}"
        options.append(pin_constraint_arg)

    # Routing graph XML:
    rr_graph = find_single_file(chip, 'fpga', part_name, 'file', 'graphfile',
                                step=None, index=None,
                                file_not_found_msg="VPR RR Graph not found")
    if (rr_graph is None):
        chip.logger.info("No VPR RR graph file specified")
        chip.logger.info("Routing architecture will come from architecture XML file")
    else:
        options.append("--read_rr_graph " + rr_graph)

    # ***NOTE: For real FPGA chips you need to specify the routing channel
    #          width explicitly.  VPR requires an explicit routing channel
    #          with when --read_rr_graph is used (typically the case for
    #          real chips).  Otherwise VPR performs a binary search for
    #          the minimum routing channel width that the circuit fits in.
    # Given the above, it may be appropriate to couple these variables somehow,
    # but --route_chan_width CAN be used by itself.
    num_routing_channels = chip.get('fpga', part_name, 'var', 'channelwidth')

    if (len(num_routing_channels) == 0):
        chip.error("Number of routing channels not specified", fatal=True)
    elif (len(num_routing_channels) == 1):
        options.append("--route_chan_width " + num_routing_channels[0])
    elif (len(num_routing_channels) > 1):
        chip.error("Only one routing channel width argument can be passed to VPR", fatal=True)

    return options


################################
# Wrapper around find files to
# help with error checking that
# only a single file is found
################################

def find_single_file(chip, *keypath, step=None, index=None, file_not_found_msg="File not found"):

    if chip.valid(*keypath) and chip.get(*keypath):
        file_list = chip.find_files(*keypath, step=step, index=index)
    else:
        file_list = []

    if (len(file_list) == 1):
        return file_list[0]
    elif (len(file_list) == 0):
        chip.logger.info(file_not_found_msg)
        return None
    else:
        chip.error("Only one file of this type can be passed to VPR", fatal=True)


################################
# Version Check
################################


def parse_version(stdout):

    # Example output of vpr --version:
    # Note that blank comment lines in this example
    # represent newlines printed by vpr --version

    # VPR FPGA Placement and Routing.
    # Version: 8.1.0-dev+c4156f225
    # Revision: v8.0.0-7887-gc4156f225
    # Compiled: 2023-06-14T17:32:05
    # Compiler: GNU 9.4.0 on Linux-5.14.0-1059-oem x86_64
    # Build Info: release IPO VTR_ASSERT_LEVEL=2
    #
    # University of Toronto
    # verilogtorouting.org
    # vtr-users@googlegroups.com
    # This is free open source code under MIT license.
    #
    #
    return stdout.split()[6]


def normalize_version(version):
    if '-' in version:
        return version.split('-')[0]
    else:
        return version


def auto_constraints():
    return 'inputs/sc_constraints.xml'


def vpr_post_process(chip):
    step = chip.get('arg', 'step')
    index = chip.get('arg', 'index')

    if os.path.exists('packing_pin_util.rpt'):
        shutil.move('packing_pin_util.rpt', 'reports')

    part_name = chip.get('fpga', 'partname')
    dff_cells = chip.get('fpga', part_name, 'resources', 'registers')
    brams_cells = chip.get('fpga', part_name, 'resources', 'brams')
    dsps_cells = chip.get('fpga', part_name, 'resources', 'dsps')

    stat_extract = re.compile(r'  \s*(.*)\s*:\s*([0-9]+)')
    lut_match = re.compile(r'([0-9]+)-LUT')
    route_length = re.compile(r'	Total wirelength: ([0-9]+)')
    log_file = f'{step}.log'
    mdata = {
        "registers": 0,
        "luts": 0,
        "dsps": 0,
        "brams": 0
    }
    with open(log_file, 'r') as f:
        in_stats = False
        for line in f:
            if in_stats:
                if not line.startswith("  "):
                    in_stats = False
                    continue
                data = stat_extract.findall(line)
                if data:
                    dtype, value = data[0]
                    dtype = dtype.strip()
                    value = int(value)

                    if dtype == "Blocks":
                        chip._record_metric(step, index, "cells", value, log_file)
                    elif dtype == "Nets":
                        chip._record_metric(step, index, "nets", value, log_file)
                    elif dtype in dff_cells:
                        mdata["registers"] += value
                    elif dtype in dsps_cells:
                        mdata["dsps"] += value
                    elif dtype in brams_cells:
                        mdata["brams"] += value
                    else:
                        lut_type = lut_match.findall(dtype)
                        if lut_type:
                            if int(lut_type[0]) == 0:
                                pass
                            else:
                                mdata["luts"] += value
            else:
                if line.startswith("Circuit Statistics:"):
                    in_stats = True
                route_len_data = route_length.findall(line)
                if route_len_data:
                    # Fake the unit since this is meaningless for the FPGA
                    units = chip.get('metric', 'wirelength', field='unit')
                    chip._record_metric(step, index, 'wirelength',
                                        int(route_len_data[0]),
                                        log_file,
                                        source_unit=units)

    for metric, value in mdata.items():
        chip._record_metric(step, index, metric, value, log_file)

    if os.path.exists(__block_file):
        with open(__block_file, 'r') as f:
            data = json.load(f)

            if "num_nets" in data and chip.get('metric', 'nets', step=step, index=index) is None:
                chip._record_metric(step, index, "nets", int(data["num_nets"]), __block_file)

            io = 0
            if "input_pins" in data:
                io += int(data["input_pins"])
            if "output_pins" in data:
                io += int(data["output_pins"])

            chip._record_metric(step, index, "pins", io, __block_file)


##################################################
if __name__ == "__main__":

    chip = make_docs()
    chip.write_manifest("vpr.json")
