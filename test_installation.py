


import logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')


logging.info("FRUIT CLASSIFIER - INSTALLATION TEST")

# Test Python version
version = sys.version_info
logging.info(f"Python {version.major}.{version.minor}.{version.micro}")
if version.major >= 3 and version.minor >= 7:
    logging.info("Python version is compatible")
else:
    logging.error("Python 3.7+ required!")
    sys.exit(1)

required_packages = {
    'numpy': 'NumPy',
    'cv2': 'OpenCV (opencv-python)',
    'sklearn': 'scikit-learn',
    'PIL': 'Pillow',
    'matplotlib': 'Matplotlib',
    'tkinter': 'Tkinter (GUI framework)'
}
failed = []
for package, name in required_packages.items():
    try:
        if failed:
            logging.error("INSTALLATION INCOMPLETE")
            logging.error("Missing packages: " + ", ".join([required_packages[pkg] for pkg in failed]))
            if 'tkinter' in failed:
                logging.error("For Tkinter: Windows/Mac: Usually included with Python | Linux: sudo apt-get install python3-tk")
            other_packages = [p for p in failed if p != 'tkinter']
            if other_packages:
                logging.error("For other packages: pip install -r requirements.txt")
                logging.error("Or install individually:")
                if 'numpy' in failed:
                    logging.error("pip install numpy")
                if 'cv2' in failed:
                    logging.error("pip install opencv-python")
                if 'sklearn' in failed:
                    logging.error("pip install scikit-learn")
                if 'PIL' in failed:
                    logging.error("pip install Pillow")
                if 'matplotlib' in failed:
                    logging.error("pip install matplotlib")
        else:
            logging.info("ALL TESTS PASSED! Your system is ready to run the Fruit Classifier.")
            logging.info("To start the application, run: python main.py")
            logging.info("To create a sample dataset for testing, run: python create_sample_dataset.py")
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

logging.info("Package versions:")
try:
    import numpy as np
    logging.info(f"NumPy: {np.__version__}")
except Exception:
    pass
try:
    import cv2
    logging.info(f"OpenCV: {cv2.__version__}")
except Exception:
    pass
try:
    import sklearn
    logging.info(f"scikit-learn: {sklearn.__version__}")
except Exception:
    pass
try:
    import PIL
    logging.info(f"Pillow: {PIL.__version__}")
except Exception:
    pass
try:
    import matplotlib
    logging.info(f"Matplotlib: {matplotlib.__version__}")
except Exception:
    pass
