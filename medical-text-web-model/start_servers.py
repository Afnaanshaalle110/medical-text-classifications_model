#!/usr/bin/env python3
"""
Startup script to run both FastAPI and Flask servers together
"""

import subprocess
import sys
import time
import threading
import signal
import os

def run_fastapi():
    """Run FastAPI server"""
    print("🚀 Starting FastAPI server on port 8000...")
    cmd = [sys.executable, "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
    return subprocess.Popen(cmd, cwd=os.getcwd())

def run_flask():
    """Run Flask server"""
    print("🌐 Starting Flask server on port 5010...")
    cmd = [sys.executable, "app.py"]
    return subprocess.Popen(cmd, cwd=os.getcwd())

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    print("\n\n🛑 Shutting down servers...")
    sys.exit(0)

def main():
    print("=" * 60)
    print("🏥 Medical Text Classification - Development Server")
    print("=" * 60)
    
    # Start FastAPI backend
    fastapi_process = None
    flask_process = None
    
    try:
        # Register signal handler for graceful shutdown
        signal.signal(signal.SIGINT, signal_handler)
        
        # Start FastAPI server
        fastapi_process = run_fastapi()
        time.sleep(3)  # Give FastAPI time to start
        
        # Start Flask server
        flask_process = run_flask()
        time.sleep(2)  # Give Flask time to start
        
        print("\n" + "=" * 60)
        print("✅ Both servers are running!")
        print("=" * 60)
        print("🔗 Frontend (Flask):  http://localhost:5010")
        print("🤖 Backend (FastAPI): http://localhost:8000")
        print("📚 API Docs:          http://localhost:8000/docs")
        print("=" * 60)
        print("🔧 Development URLs:")
        print("   • Home Page:       http://localhost:5010/")
        print("   • Classifier:      http://localhost:5010/classifier")
        print("   • Specialties:     http://localhost:5010/specialties")
        print("   • About:           http://localhost:5010/about")
        print("=" * 60)
        print("Press Ctrl+C to stop both servers")
        print("=" * 60)
        
        # Wait for both processes to complete
        while True:
            if fastapi_process.poll() is not None:
                print("❌ FastAPI server stopped unexpectedly")
                break
            if flask_process.poll() is not None:
                print("❌ Flask server stopped unexpectedly")
                break
            time.sleep(1)
    
    except KeyboardInterrupt:
        print("\n🛑 Received interrupt signal")
    
    finally:
        # Clean up processes
        if fastapi_process:
            print("🔄 Stopping FastAPI server...")
            fastapi_process.terminate()
            try:
                fastapi_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                fastapi_process.kill()
        
        if flask_process:
            print("🔄 Stopping Flask server...")
            flask_process.terminate()
            try:
                flask_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                flask_process.kill()
        
        print("✅ All servers stopped. Goodbye!")

if __name__ == "__main__":
    main() 