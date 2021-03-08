# Copyright 2020 Silicon Compiler Authors. All Rights Reserved.

import math
import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

class Floorplan:
    """
    Chip Layout Class
    """
    def __init__(self):
        '''
        Init method for Chip object
        '''
        layout = {} 
        layout["version"] = "5.8"
        layout["dividerchar"] = "/"
        layout["busbitchar"] = "[]"
        layout["design"] = ""
        layout["diearea"] = []
        layout["track"] = {}
        layout["component"] = {}
        layout["pin"] = {}
        layout["net"] = {}
        layout["net"] = {}
            
    def writejson(self, filename):
        '''
        Write JSOn File
        '''
        logging.info('Write JSON %s', filename)
        pass

    def writedef(self, filename):
        '''
        Write DEF File
        '''
        logging.info('Write DEF %s', filename)
        pass

    def diearea(self, *args):
        '''
        Specifies the die area of the design. If two points are defined, 
        specifies two corners of the bounding rectangle for the design. If
        more than two points are defined, specifies the points of a 
        polygon that forms the die area. All points are integers, 
        specified as DEF database units.
        '''
        logging.info('Adding diearea to floorplan')
        pass

    def addrow(self, site, x, y, orientation, numx, numy, stepx, stepy):
        '''
        Adding a placement row
        '''
        logging.info('Adding  row to floorplan')
        pass

    def addtrack(self, layer, direction, start, step, total):
        '''
        Add a routing track
        '''
        logging.info('Placing tracks')
        pass
    
    
    def placepin(self, pin, net, x, y, box, layer,
               direction=None, use="signal", fixed=True):
        '''
        Place pin
        '''
        logging.info('Placing a pin')
        pass

    def placecell(self):
        '''
        Place a cell
        '''
        logging.info('Placing a cell')
        pass

    def placekeepout(self):
        '''
        Place a keepout area
        '''
        logging.info('Placing a keepout area')
        pass

    

#############################
#Snaps value to routing grid
def snap2grid (val, grid):
    logging.debug('Executing fp.snap2grid', val,grid)
    return(grid * math.ceil(val/grid))

#####################
# Place Blockage
def blockage (name, box, metal):
    logging.debug('Executing fp.cell', name, x, y, orientation)

#####################
#Place a list of pins
#Everything starts in the lower left corer (0,0) and counts up and to the right

def pinlist (pinlist, side, block_w, block_h, offset, pinwidth, pindepth, pinhalo, pitch, metal):

    logging.debug('Executing pinlist',pinlist, side, block_w, block_h, offset, pinwidth, pindepth, pinhalo, pitch, metal)
    
    if(side=='no'):
        x0    = offset
        y0    = block_h - halo
        x1    = x0 + pinwidth
        y1    = y0 - pindepth + 2 * halo
        xincr = pitch
        yincr = 0.0
    elif (side == 'so'):
        x0    = offset
        y0    = halo
        x1    = x0 + pinwidth
        y1    = y0 + pindepth - 2 * halo
        xincr = pitch
        yincr = 0.0
    elif (side == 'we'):
        x0    = halo
        y0    = offset
        x1    = x0 + pindepth - 2 * halo
        y1    = y0 + pinwidth
        xincr = 0.0 
        yincr = pitch
    elif (side == 'ea'):
        x0    = block_w - halo
        y0    = offset
        x1    = x0 - pindepth + 2 * halo
        y1    = y0 + pinwidth
        xincr = 0.0
        yincr = pitch    
    #loop through all pins
    for name in pinlist:
        box = [x0,y0,x1,y1]
        pin (name, box, metal)
        #Update with new values
        x0 = x0 + xincr
        y0 = y0 + yincr
        x1 = x1 + xincr
        y1 = y1 + yincr



#####################
# Plot the design
# Input=structure
# Output=display | write to file

def plot():
    logging.debug('Display design')

    design = "Floorplan"
    width  = 1000
    height = 1000
    
    fig, ax = plt.subplots()
    ax.tick_params(labeltop=True,
                   top=True,
                   right=True,
                   labelright=True)   # Put labels on both side for ease uf use
    ax.grid(True)                     # Turn on grids
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_xlabel('Width')            # x label
    ax.set_ylabel('Height')           # y label
    ax.set_title(design)              # Name of design
    
    for i in range(10):
        for j in range(10):
            print(i*100, j*100)
            ax.add_patch(Rectangle((i*100,j*100), 50, 50,
                           edgecolor = 'pink',
                           facecolor = 'blue',
                           fill=True,
                           lw=1))
    plt.savefig("test.svg", format="svg")
    plt.show()                        # Display figure




#####################
# Test Program
    
#plot()
   
