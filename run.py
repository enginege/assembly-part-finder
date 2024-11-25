import sys
import os

# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from retrieval_system.main import main

if __name__ == "__main__":
    main()