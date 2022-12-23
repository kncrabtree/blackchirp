Installation
============

Requirements
............

Blackchirp is a cross-platform program, though at present it has only been tested on the Linux distribution openSUSE 15.2+.
If the program does not behave as expected on other platforms, please `raise an issue`_.

The prerequisites for compiling Blackchirp are:

- Compiler with C++17 support (e.g., gcc 6+)
- Qt 5.15+ and < 6 (Qt 6.2 may work, but is untested. Qt 6.0 and 6.1 are not compatible with Blackchirp)
- `qwt`_ 6.1+
- `Eigen template library`_ v3+
- `GNU Scientific Library`_ v2.1+
- (optional) `CUDA`_ v9+ with capable GPU

.. _raise an issue: https://github.com/kncrabtree/blackchirp/issues
.. _qwt: https://qwt.sourceforge.io/
.. _Eigen template library: https://eigen.tuxfamily.org/index.php?title=Main_Page
.. _GNU Scientific Library: https://www.gnu.org/software/gsl/
.. _CUDA: https://developer.nvidia.com/cuda-downloads

The easist way to build Blackchirp is to open the ``blackchirp.pro`` file in the `Qt Creator`_ IDE, which allows you to easily configure a qmake/C++ kit and control whether the build includes debugging symbols.
It is recommended to build Blackchirp in the release configuration for best performance.

.. _Qt Creator: https://www.qt.io/product/development-tools

Configuration
.............

The user-controllable build parameters are specified in ``src/config/config.pri``.
This file does not exist in the repository, as the contents will be different for every user, so a template file is provided as ``src/config/config.pri.template``.
Simply copy that template to ``src/config/config.pri`` and it can then be edited and used to build the program.
By editing this file, there are three main aspects of the program you can control: optional modules, hardware implementations, and custom library/include paths.

Optional Modules
----------------

As of Blackchirp v1.0, there are two optional modules that may be enabled by adding the following lines to ``config.pri``:

- ``CONFIG += gpu-cuda`` enables use of a CUDA-capable GPU for performing FID averaging. This offers a slight performance improvement if the number of data points acquired in a single shot is large (over 1,000,000) or if only a small number of CPU cores are available. To successfully compile the CUDA code in Blackchirp, you need to add the CUDA compiler to the qmake build. An example is provided in config.pri.template which works with default packages in openSUSE 15+.

- ``CONFIG += lif`` enables simultaneous acquitision of CP-FTMW data and Laser-induced fluorescence data (as of v1.0-alpha, LIF support is not enabled). Support for LIF hardware and functionality is minimal at present, as it is anticipated that this module will see little use outside of UC Davis.

Leaving these lines out (or commenting them out) will remove the associated code from the build.
First-time users are advised to leave them out.

Hardware Implementations
------------------------

In Blackchirp, each piece of hardware is made up of two components: the base type of hardware that describes its function within the instrument, and an implementation that corresponds to a specific manufacturer/model.
The ``config.pri`` file allows you to select which specific implementations of hardware you have, as well as which pieces of hardware your instrument has (or those you wish Blackchirp to communicate with and/or control).
For each hardware type, the associated implementation is chosen by assigning a number as shown in the ``config.pri.template`` file.
For example, if your FTMW digitizer is a Tektronix DSA71604C, you would set ``FTMWSCOPE=1``.

All hardware items have a "virtual" implementation (number ``0``) that can serve as a placeholder for a physical device.
While this is primarily intended for development purposes, it can also be useful in some applications.
For example, if you do not have a flow controller setup, you can compile Blackchirp with a virtual flow controller (``FC=0``) and enter the names of the sample gases you are using.
Blackchirp will record these values to disk during each experiment.

The only items of hardware that are required for Blackchirp to run are a FTMW digitizer (``FTMWSCOPE``) and at least one "Clock" (``CLOCKS``), which Blackchirp uses to record your specific upconversion/downconversion chain if necessary.
Unlike the other hardware items, Blackchirp supports up to 5 clocks, and these are specified by entering up to 5 numbers separated by spaces.
The "virtual" implementation (``0``) is called a ``FixedClock``, and it contains 5 virtual "outputs" that can be assigned different frequencies.
More information about clocks and setting up your RF configuration can be found in the User Guide (TODO: add link).

Custom Library/Include Paths
----------------------------

To compile Blackchirp, you must provide the appropriate libraries: qwt, gsl, m, gslcblas, and any other libraries needed for your hardware implementations.
In ``config.pri.template``, an example is provided for Linux assuming these libraries exist in your ``LD_LIBRARY_PATH`` (usually ``/usr/lib``, ``/usr/lib64``, and ``/usr/local/lib``).
If your libraries are located elsewhere, you will need to add them.
Using Qt Creator, you can right-click inside the `config.pri` file and use the "Add Library" option, or you can consult the `qmake documentation`_ for details.

.. _qmake documentation: https://doc.qt.io/qt-5/qmake-variable-reference.html#libs

Finally, you will need to ensure that the compiler can find the header files for qwt, eigen3, and gsl.
Blackchirp's source code assumes that include files for each of these libraries is found within a subfolder of your ``INCLUDE`` path: ``qwt6``, ``eigen3``, and ``gsl`` respectively.
If this is not already the case on your system, you can create symbolic links and modify the qmake ``INCLUDEPATH`` variable as needed.
