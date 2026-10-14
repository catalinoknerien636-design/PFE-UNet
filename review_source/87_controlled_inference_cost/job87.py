from pathlib import Path
import sys
R=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(R/'85_device_lifecycle_qualification'))
from job85 import w,LifecycleJob,reason
