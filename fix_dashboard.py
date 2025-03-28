#!/usr/bin/env python3
"""
Simple script to start the dashboard with debug mode enabled
"""

import os
import sys
import subprocess
import logging
import time
import signal
import shutil
import threading # Use threading to read output without blocking

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_command(cmd, cwd=None):
    logger.info(f"Running command: {cmd}")
    
    try:
        process = subprocess.Popen(
            cmd,
            shell=True, 
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            cwd=cwd
        )
        
        for line in process.stdout:
            print(line.strip())
        
        process.wait()
        return process.returncode
    except Exception as e:
        logger.error(f"Error running command: {e}")
        return -1

def find_node_executable():
    # First check if node is in PATH
    node_path = shutil.which('node')
    if node_path:
        return node_path
    
    # If not in PATH, try common locations
    common_paths = [
        '/usr/local/bin/node',
        '/usr/bin/node',
        '/opt/local/bin/node',
        '/opt/homebrew/bin/node',
    ]
    
    for path in common_paths:
        if os.path.exists(path):
            return path
    
    return None

# Function to continuously read and print output from a process
def stream_output(process, log_prefix=""):
    try:
        for line in iter(process.stdout.readline, ''):
            if line:
                logger.info(f"{log_prefix}: {line.strip()}")
        process.stdout.close()
    except Exception as e:
        logger.error(f"Error reading output for {log_prefix}: {e}")

def main():
    project_root = os.path.dirname(os.path.abspath(__file__))
    react_app_dir = os.path.join(project_root, 'app/frontend/react-app')
    
    logger.info(f"Project root: {project_root}")
    logger.info(f"React app directory: {react_app_dir}")
    
    # Check if node is installed
    node_path = find_node_executable()
    if not node_path:
        logger.error("Node.js executable not found. Please install Node.js.")
        return 1
    
    logger.info(f"Found Node.js executable: {node_path}")
    
    # Start the backend Python FastAPI server
    backend_server_script = os.path.join(project_root, 'app/frontend/server.py')
    logger.info(f"Starting backend server: {backend_server_script}")
    
    python_cmd = sys.executable
    backend_cmd = f"{python_cmd} {backend_server_script}"
    backend_process = subprocess.Popen(
        backend_cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1 # Line buffered
    )
    # Start thread to read backend output
    backend_thread = threading.Thread(target=stream_output, args=(backend_process, "Backend"))
    backend_thread.daemon = True
    backend_thread.start()

    start_process = None # Initialize start_process
    start_thread = None

    try:
        # Wait for backend to initialize (adjust time if needed)
        logger.info("Waiting for backend to initialize (approx 10s)...")
        time.sleep(10) # Increased wait time
        
        # Check if backend is still running
        if backend_process.poll() is not None:
             logger.error("Backend server failed to start. Check backend logs.")
             return 1

        # Start React development server
        logger.info(f"Starting React development server in {react_app_dir}")
        
        env = os.environ.copy()
        env['DISABLE_ESLINT_PLUGIN'] = 'true'
        env['TSC_COMPILE_ON_ERROR'] = 'true'
        env['BROWSER'] = 'none'  # Don't open browser automatically
        # Add PORT if you want to force it, otherwise it defaults to 3000
        # env['PORT'] = '3000' 
        
        start_cmd = "npm start"
        start_process = subprocess.Popen(
            start_cmd,
            shell=True,
            cwd=react_app_dir,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1 # Line buffered
        )
        # Start thread to read frontend output
        start_thread = threading.Thread(target=stream_output, args=(start_process, "Frontend"))
        start_thread.daemon = True
        start_thread.start()
        
        logger.info("React dev server process started. Check logs for 'Starting the development server...'")
        
        # Wait for processes indefinitely until one exits or Ctrl+C
        while True:
             if backend_process.poll() is not None:
                 logger.warning("Backend server exited.")
                 break
             if start_process.poll() is not None:
                 logger.warning("React dev server exited.")
                 break
             time.sleep(1)
             
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down...")
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
    finally:
        # Clean up processes
        logger.info("Initiating shutdown...")
        if start_process and start_process.poll() is None:
            logger.info("Terminating React server...")
            start_process.terminate()
            try:
                start_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning("React server did not terminate gracefully, killing.")
                start_process.kill()
            
        if backend_process.poll() is None:
            logger.info("Terminating backend server...")
            backend_process.terminate()
            try:
                backend_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning("Backend server did not terminate gracefully, killing.")
                backend_process.kill()

        # Wait for threads to finish
        if backend_thread and backend_thread.is_alive():
             backend_thread.join(timeout=2)
        if start_thread and start_thread.is_alive():
             start_thread.join(timeout=2)
             
        logger.info("Shutdown complete.")
    
    return 0

if __name__ == "__main__":
    # Ensure correct signal handling for graceful shutdown
    signal.signal(signal.SIGINT, lambda s, f: sys.exit(0))
    signal.signal(signal.SIGTERM, lambda s, f: sys.exit(0))
    sys.exit(main()) 