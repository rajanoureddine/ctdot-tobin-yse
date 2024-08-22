"""
At some point I got tired of setting the paths manually for every file. 
But this approach didn't last long... 
"""

import pathlib

cwd = pathlib.Path().resolve().parent
data_dir = pathlib.Path().resolve().parent / "Documents" / "tobin_working_data"