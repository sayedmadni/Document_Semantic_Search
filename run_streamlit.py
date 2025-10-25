#!/usr/bin/env python3
"""
Launcher script for the Document Semantic Search Streamlit app.
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    """Launch the Streamlit app."""
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    
    # Path to the streamlit app
    app_path = script_dir / "streamlit_app.py"
    
    if not app_path.exists():
        print(f"Error: Streamlit app not found at {app_path}")
        sys.exit(1)
    
    # Change to the project directory
    os.chdir(script_dir)
    
    # Run streamlit
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            str(app_path),
            "--server.port", "8501",
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false"
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running Streamlit: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nStreamlit app stopped.")

if __name__ == "__main__":
    main()
