import importlib
import os
import siliconcompiler

####################################################
# EDA Setup
####################################################
def setup_eda(chip, name=None):
    chip.logger.debug("Setting up EDA 'eda_default.py'")     

    # Define Compilation Flow
    chip.cfg['steps']['value'] = ['import',
                                  'syn',
                                  'floorplan',
                                  'place',
                                  'cts',
                                  'route',
                                  'dfm',
                                  'export']
    

    chip.cfg['start']['value'] = ['import']
    chip.cfg['stop']['value'] = ['export']
        
    # Setup tool based on flow step
    for step in chip.cfg['steps']['value']:            
        if step == 'import':
            vendor = 'verilator'
        elif step == 'syn':
            vendor = 'yosys'   
        elif step == 'export':
            vendor = 'klayout'
        else:
            vendor = 'openroad'
            
        #load module dynamically on each step
        #see sys.path

        packdir = "eda." + vendor    
        module = importlib.import_module('.'+vendor, package=packdir)
        setup_tool = getattr(module,'setup_tool')
        setup_tool(chip, step)
        
#########################
if __name__ == "__main__":    

    # File being executed
    prefix = os.path.splitext(os.path.basename(__file__))[0]
    output = prefix + '.json'

    # create a chip instance
    chip = siliconcompiler.Chip()
    # load configuration
    setup_eda(chip, 'syn')
    # write out result
    chip.writecfg(output)