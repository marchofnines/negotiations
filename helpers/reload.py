"""
Author: Basil Haddad
Date: 11.01.2023

Description:
    Contains just one function myreload to help reload libraries from Jupyter Notebooks
"""

from importlib import reload
 
def myreload():
    """
    Reload specific modules from the 'helpers' package.
    
    Define variables for modules as Global Variables so they can be accessed outside this function:
    - reimport: Reload helpers.my_imports module
    - xfrs: Reload helpers.transformers.py module
    - pp: Reload helpers.preprocessing module
    - plot: Reload helpers.plot module
    - tools: Reload helpers.tools module
    
    Prints:
    - Confirmation message indicating which modules were reloaded.
    """
    # Declare variables for modules as global so they can be accessed outside this function
    global reimport, xfrs, pp, plot, tools  # Declare as global variables
    # Import the specific modules from the 'helpers' package
    import helpers.my_imports as reimport
    import helpers.transformers as xfrs
    import helpers.preprocessing as pp
    import helpers.plot as plot
    import helpers.tools as tools
    # Reload modules
    reload(reimport)
    reload(xfrs)
    reload(pp)
    reload(plot)
    reload(tools)
    print("Reloaded helpers.preprocessing, helpers.plots, and helpers.tools.")
