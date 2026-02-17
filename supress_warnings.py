import sys
import os
import ctypes

def suppress_cv_warnings():
    """Suppress OpenCV warnings on Windows"""
    # Method 1: Redirect stderr
    if sys.platform == 'win32':
        # Get the Windows null device
        DEVNULL = open(os.devnull, 'w')
        # Save original stderr
        old_stderr = sys.stderr
        # Redirect stderr during imports
        sys.stderr = DEVNULL
    
    # Set environment variables
    os.environ['OPENCV_LOG_LEVEL'] = 'SILENT'
    os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '0'
    
    return old_stderr if sys.platform == 'win32' else None

def restore_stderr(old_stderr):
    """Restore stderr after cv2 import"""
    if old_stderr is not None:
        sys.stderr = old_stderr