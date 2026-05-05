import platform
import psutil

print("CPU:", platform.processor())
print("Machine:", platform.machine())
print("OS:", platform.system(), platform.release())
print("Python Version:", platform.python_version())

# RAM
ram = psutil.virtual_memory()
print("Total RAM (GB):", round(ram.total / (1024**3), 2))