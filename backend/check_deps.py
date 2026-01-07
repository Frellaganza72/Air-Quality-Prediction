import sys
import importlib.util

REQUIRED_PACKAGES = [
    ('flask', 'Flask'),
    ('flask_cors', 'Flask-CORS'),
    ('pandas', 'Pandas'),
    ('numpy', 'NumPy'),
    ('sklearn', 'Scikit-Learn'),
    ('joblib', 'Joblib'),
    ('dotenv', 'Python-Dotenv'),
    ('requests', 'Requests'),
    ('scipy', 'SciPy'),
    ('apscheduler', 'APScheduler'),
    ('tensorflow', 'TensorFlow')
]

def check_dependencies():
    missing = []
    print("🔍 Checking Python dependencies...")
    
    for package_name, display_name in REQUIRED_PACKAGES:
        try:
            if importlib.util.find_spec(package_name) is None:
                missing.append(display_name)
        except ImportError:
            missing.append(display_name)
        except Exception as e:
            # Handle other potential errors during import check
            print(f"⚠️ Error checking {display_name}: {e}")
            missing.append(display_name)
    
    if missing:
        print(f"❌ Missing dependencies: {', '.join(missing)}")
        return False
    
    print("✅ All dependencies satisfied.")
    return True

if __name__ == "__main__":
    if not check_dependencies():
        sys.exit(1)
    sys.exit(0)
