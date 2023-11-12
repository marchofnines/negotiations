import sys
sys.path.append('/Users/basilhaddad/jupyter/capstone/')
from importlib import reload

    
def myreload():
    """
    Reload specific modules from the 'helpers' package.
    
    Define variables for modules as Global Variables so they can be accessed outside this function:
    - reimport: Reloaded 'helpers.my_imports' module
    - pp: Reloaded 'helpers.preprocessing' module
    - plot: Reloaded 'helpers.plot' module
    - tools: Reloaded 'helpers.tools' module
    
    Prints:
    - Confirmation message indicating which modules were reloaded.
    """
    # Declare variables for modules as global so they can be accessed outside this function
    global reimport, pp, plot, tools  # Declare as global variables
    # Import the specific modules from the 'helpers' package
    import helpers.my_imports as reimport
    import helpers.preprocessing as pp
    import helpers.plot as plot
    import helpers.tools as tools
    # Reload modules
    reload(reimport)
    reload(pp)
    reload(plot)
    reload(tools)
    print("Reloaded helpers.preprocessing, helpers.plots, and helpers.tools.")
