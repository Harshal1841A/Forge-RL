import sys
try:
    import app
    print("App imported successfully")
except Exception as e:
    import traceback
    traceback.print_exc()
    sys.exit(1)
