"""
Test Installation Script
Run this to verify all dependencies are installed correctly
"""

import sys

print("="*60)
print("  FRUIT CLASSIFIER - INSTALLATION TEST")
print("="*60)
print()

# Test Python version
print("1. Testing Python version...")
version = sys.version_info
print(f"   Python {version.major}.{version.minor}.{version.micro}")
if version.major >= 3 and version.minor >= 7:
    print("   ✓ Python version is compatible")
else:
    print("   ❌ Python 3.7+ required!")
    sys.exit(1)
print()

# Test required packages
required_packages = {
    'numpy': 'NumPy',
    'cv2': 'OpenCV (opencv-python)',
    'sklearn': 'scikit-learn',
    'PIL': 'Pillow',
    'matplotlib': 'Matplotlib',
    'tkinter': 'Tkinter (GUI framework)'
}

print("2. Testing required packages...")
failed = []

for package, name in required_packages.items():
    try:
        if package == 'tkinter':
            import tkinter
        else:
            __import__(package)
        print(f"   ✓ {name}")
    except ImportError:
        print(f"   ❌ {name} - NOT INSTALLED")
        failed.append(package)

print()

if failed:
    print("="*60)
    print("  INSTALLATION INCOMPLETE")
    print("="*60)
    print()
    print("Missing packages:")
    for pkg in failed:
        print(f"  • {required_packages[pkg]}")
    print()
    print("To install missing packages, run:")
    print()
    
    if 'tkinter' in failed:
        print("  For Tkinter:")
        print("    Windows/Mac: Usually included with Python")
        print("    Linux: sudo apt-get install python3-tk")
        print()
    
    other_packages = [p for p in failed if p != 'tkinter']
    if other_packages:
        print("  For other packages:")
        print("    pip install -r requirements.txt")
        print()
        print("  Or install individually:")
        if 'numpy' in failed:
            print("    pip install numpy")
        if 'cv2' in failed:
            print("    pip install opencv-python")
        if 'sklearn' in failed:
            print("    pip install scikit-learn")
        if 'PIL' in failed:
            print("    pip install Pillow")
        if 'matplotlib' in failed:
            print("    pip install matplotlib")
    
    print()
    print("="*60)
else:
    print("="*60)
    print("  ✓ ALL TESTS PASSED!")
    print("="*60)
    print()
    print("Your system is ready to run the Fruit Classifier!")
    print()
    print("To start the application, run:")
    print("  python main.py")
    print()
    print("To create a sample dataset for testing, run:")
    print("  python create_sample_dataset.py")
    print()
    print("="*60)

# Additional information
print()
print("Package versions:")
try:
    import numpy as np
    print(f"  NumPy: {np.__version__}")
except:
    pass

try:
    import cv2
    print(f"  OpenCV: {cv2.__version__}")
except:
    pass

try:
    import sklearn
    print(f"  scikit-learn: {sklearn.__version__}")
except:
    pass

try:
    import PIL
    print(f"  Pillow: {PIL.__version__}")
except:
    pass

try:
    import matplotlib
    print(f"  Matplotlib: {matplotlib.__version__}")
except:
    pass

print()
print("="*60)
