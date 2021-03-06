OpenCV for Embedded Product Testing
====================================

Welcome to the OpenCV for Embedded Product Testing project.  This project 
demonstrates how computer vision can be used to test embedded 
software products, and the project includes three examples:

* Finding an icon in an image.
* Detect when an LED is illuminated.
* Measure the latency of a touchscreen.

This project is written in Python 2.7 and relies upon OpenCV, numpy and scipy.  
`OpenCV <http://opencv.org/>`_ is the Open Source Computer Vision library which 
includes excellent Python bindings.  The Python bindings bring the performance 
advantage of native C/C++ code with the flexibility of Python.  
`NumPy <http://www.numpy.org/>`_ is a scientific computing package for Python.
`SciPy <http://www.scipy.org/>`_ builds upon NumPy to provide broad support
for solving math, science and engineering problems.  

All software development was performed under Ubuntu, but the project should run
under Windows and Mac OS X.  Some features require use of the nonfree library
which may not be included through the some Linux distribution package managers,
including Ubuntu.

For Windows installations, this project is tested using the 
`WinPython 2.7.6.3 <http://winpython.sourceforge.net/>`_ 
distribution with the 
`opencv-python package <http://www.lfd.uci.edu/~gohlke/pythonlibs/#opencv>`_.  
Download both files and execute the 
WinPython installer.  Once WinPython installs, run 
"WinPython Control Panel.exe", click "Add packages", select the opencv-python 
package, and then click "Install Packages".  You will likely also want to
select "Advanced"->"Register Distribution...".  

The project is released under the permissive MIT license.
