.. Scripts


Scripts
=======

All scripts are located in the bin directory.  To display help for a script, 
type bin/[script_name] --help

The available scripts are:

* find_icon_exact.py: Find an icon in an image with exacting pixel accuracy.
* find_icon.py: Find an icon in an image while allowing variation in the exact
  pixels.  This method allows icons to match in images capture through cameras
  and webcams.
* detect_led.py: Detect when an LED is illuminated on a product or PCB.  The
  script allows the user to selected the product, select the LED location and
  then monitors when the LED is illuminated.
* measure_touchscreen_latency.py: Measure the time between the user's finger 
  motion and the corresponding motion on the touchscreen display.
  The test requires software running on the target device that displays a white
  background and magenta dot.

The bin/measure_touchscreen_latency.py script works with the Android application
"Touch Latency" which is included in this project.  The TouchLatency directory
contains an Android application project that can be built with the Android SDK.

