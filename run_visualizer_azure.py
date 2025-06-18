#!/usr/bin/env python3
"""
Azure deployment script for the Clinic ZIP Code Visualizer
Run this script on your Azure VM to start the application with public access
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ Successfully installed requirements")
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install requirements: {e}")
        sys.exit(1)

def run_streamlit():
    """Run Streamlit with Azure-appropriate settings"""
    print("Starting Streamlit application...")
    print("The application will be accessible from any IP address on port 8502")
    print("Make sure port 8502 is open in your Azure Network Security Group")
    
    try:
        # Run streamlit with host 0.0.0.0 to allow external connections
        subprocess.run([
            "streamlit", "run", "visualizer.py",
            "--server.address", "0.0.0.0",
            "--server.port", "8502",
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\n✓ Application stopped by user")
    except Exception as e:
        print(f"✗ Error running application: {e}")

if __name__ == "__main__":
    print("=== Clinic ZIP Code Visualizer - Azure Deployment ===")
    print()
    
    # Check if we're in the right directory
    if not os.path.exists("visualizer.py"):
        print("✗ visualizer.py not found. Make sure you're in the correct directory.")
        sys.exit(1)
    
    install_requirements()
    print()
    run_streamlit() 