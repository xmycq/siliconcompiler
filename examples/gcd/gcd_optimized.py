#!/usr/bin/env python3
# Copyright 2020 Silicon Compiler Authors. All Rights Reserved.

import os
import siliconcompiler


def main(root='.'):
    '''Simple asicflow example.'''

    chip = siliconcompiler.Chip('gcd')
    chip.input(f"{root}/gcd.v")
    chip.input(f"{root}/gcd.sdc")
    chip.set('option', 'relax', True)
    chip.set('option', 'quiet', True)
    chip.set('option', 'track', True)
    chip.set('option', 'hash', True)
    chip.set('option', 'skipcheck', True)
    chip.set('option', 'novercheck', True)
    chip.set('option', 'nodisplay', True)
    chip.set('constraint', 'outline', [(0, 0), (100.13, 100.8)])
    chip.set('constraint', 'corearea', [(10.07, 11.2), (90.25, 91)])
    chip.load_target("skywater130_demo")
    chip.optimize_parameters(
        [
            {
                "key": ['tool', 'openroad', 'task', 'place', 'var', 'place_density'],
                "values": [0.70, 0.95],
                "type": 'float'
            },
            {
                "key": ['tool', 'openroad', 'task', 'cts', 'var', 'cts_cluster_diameter'],
                "values": [10, 100],
                "type": 'int'
            }
        ],
        [
            {
                "key": ["metric", "setupslack"],
                "target": "max"
            },
            # {
            #     "key": ["metric", "holdslack"],
            #     "target": "max"
            # }
        ]
    )
    chip.summary()


if __name__ == '__main__':
    main(root=os.path.dirname(__file__))
