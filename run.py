import sys
import os
import logging
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

logging.basicConfig(level=logging.DEBUG, filename='run.log', filemode='w')

# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from retrieval_system.main import main

if __name__ == "__main__":
    main()