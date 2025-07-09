#!/usr/bin/env python3
"""
Start both Optimizer and Visualizer on Azure VM
This script runs both applications simultaneously on different ports
"""

import subprocess
import sys
import os
import threading
import time

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ Successfully installed requirements")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install requirements: {e}")
        return False

def get_streamlit_path():
    """Find the correct streamlit executable path"""
    # Try common locations
    possible_paths = [
        "/home/viriditytech/.local/bin/streamlit",
        "/usr/local/bin/streamlit", 
        "/usr/bin/streamlit"
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    # Fallback: try to find it using which
    try:
        result = subprocess.run(["which", "streamlit"], capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout.strip()
    except:
        pass
    
    # Last resort: use python -m streamlit
    return None

def run_app():
    """Run the unified Streamlit application on port 8501"""
    print("Starting unified app on port 8501...")
    try:
        streamlit_path = get_streamlit_path()
        if streamlit_path:
            cmd = [
                streamlit_path, "run", "app.py",
                "--server.address", "0.0.0.0",
                "--server.port", "8501",
                "--server.headless", "true",
                "--browser.gatherUsageStats", "false"
            ]
        else:
            cmd = [
                sys.executable, "-m", "streamlit", "run", "app.py",
                "--server.address", "0.0.0.0",
                "--server.port", "8501",
                "--server.headless", "true",
                "--browser.gatherUsageStats", "false"
            ]
        subprocess.run(cmd)
    except Exception as e:
        print(f"✗ App error: {e}")

if __name__ == "__main__":
    print("=== Clinic ZIP Code Applications - Azure Deployment ===")
    print()
    
    # Check if we're in the right directory
    missing_files = []
    if not os.path.exists("optimizer.py"):
        missing_files.append("optimizer.py")
    if not os.path.exists("visualizer.py"):
        missing_files.append("visualizer.py")
    
    if missing_files:
        print(f"✗ Missing files: {', '.join(missing_files)}")
        print("Make sure you're in the correct directory.")
        sys.exit(1)
    
    # Install requirements
    if not install_requirements():
        sys.exit(1)
    
    # Check streamlit installation
    streamlit_path = get_streamlit_path()
    if streamlit_path:
        print(f"✓ Found streamlit at: {streamlit_path}")
    else:
        print("✓ Will use python -m streamlit")
    
    print("\n" + "="*60)
    print("Starting application...")
    print("App will be available at: http://20.127.202.39:8501")
    print("Make sure ports 8501 and 8502 are open in your NSG!")
    print("Press Ctrl+C to stop both applications")
    print("="*60 + "\n")
    
    try:
        run_app()
    except KeyboardInterrupt:
        print("\n✓ Application stopped by user")
        sys.exit(0) 