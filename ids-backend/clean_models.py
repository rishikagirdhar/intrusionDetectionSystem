import os
import glob

def clean_models():
    files = glob.glob('models/*.pkl')
    for f in files:
        try:
            os.remove(f)
            print(f"Deleted: {f}")
        except Exception as e:
            print(f"Error deleting {f}: {str(e)}")

if __name__ == "__main__":
    clean_models()