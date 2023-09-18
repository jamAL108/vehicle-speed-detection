import pyopencl as cl
import cv2
platforms = cl.get_platforms()
devices = platforms[0].get_devices(cl.device_type.GPU)

for device in devices:
    print("Device Name:", device.name)
# Enable OpenCL support
cv2.ocl.setUseOpenCL(True)

# Retrieve available OpenCL devices
devices = cv2.ocl.getOpenCLDevices()

# Select the first OpenCL device (usually the GPU)
if devices:
    cv2.ocl.setOpenCLDevice(devices[0])

# Create an OpenCL context
ctx = cv2.UMatContext()

# Now, cv2.UMat objects will use the OpenCL context when applicable


