import sys
import os
import logging

logging.basicConfig(level=logging.DEBUG, filename='run.log', filemode='w')

# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from retrieval_system.main import main

if __name__ == "__main__":
    main()