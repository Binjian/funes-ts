import sys, os
import pprint

from .models import *
from .metrics import *
from .misc import *
# from src import PROJROOT
from grid_search import grid_search
from src.tgan import PROJROOT

# Set the project root directory
# Set the package directory
def main():
    pp = pprint.PrettyPrinter(indent=40)
    pp.pprint(f"tgan __main__ CWD: {os.getcwd()}")
    pp.pprint(f"PROJROOT: {tgan.PROJROOT}")



if __name__ == "__main__":
    main()
    print("start grid search of tgan")
    grid_search()
