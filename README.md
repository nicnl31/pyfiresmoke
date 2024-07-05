# ANSCV Fire/Smoke Detection
OpenCV library for ANSVIS. 

**NOTE**: the README needs revision.

# Dependencies
1. **OpenCV 4.7** (GPU support) https://storage.googleapis.com/anscenter-public-resources/ANSCENTER_RESOURCES/Utilities/opencv.zip


2. Other third-party libraries:
   - **MoveDetect**: https://github.com/stephanecharette/MoveDetect?tab=readme-ov-file
   - **Boost C++**: https://boostorg.jfrog.io/artifactory/main/release/1.85.0/source/

# Structure

- The project solution is located in C:\Projects\ANSCV folder
- OpenCV is located in C:\OpenCV folder
- Boost C++ is located in C:\Projects\Research\CPlusLibraries\boost

```
C:/
├── OpenCV
└── Projects/
    ├── ANSCV
    └── Research/
        └── CPlusLibraries/
            └── boost
```

# Detection algorithm

1. Use the MoveDetect library to detect moving objects, and get ROIs.
2. Detect fire on ROIs using HSV algorithm: github.com/gunarakulangunaretnam/fire-detection-system-in-python-opencv
3. Check if ROIs are expanding over frames (it is likely to be fire).
4. Check surrouding ROIs for smoke.
5. Give detection outcome

# Classification algorithm