Installation
============

Requirements
............

Blackchirp is a cross-platform program. It has been tested on Windows 11, Max OSX, and Linux (openSUSE Leap 15.2+, openSUSE Tumbleweed, and Ubuntu 22.04.3 LTS).
If the program does not behave as expected on other platforms, please `raise an issue`_.

The prerequisites for compiling Blackchirp are:

- Compiler with C++2b support (e.g., gcc 11+, MSVC 19.28+, clang 10+)
- `Qt 5.15+ or 6.7+ <https://www.qt.io/download-qt-installer-oss>`_. **Blackchirp will drop support for Qt5 in a future release; it is strongly recommended that you build against Qt 6.7+!**
- `qwt`_ 6.2+
- `Boost C++ Libraries <https://www.boost.org/>`_ (preprocessor library only)
- `Eigen template library`_ v3+
- `GNU Scientific Library`_ v2.1+
- (optional) `CUDA`_ v9+ with capable GPU

.. _raise an issue: https://github.com/kncrabtree/blackchirp/issues
.. _qwt: https://qwt.sourceforge.io/
.. _Eigen template library: https://eigen.tuxfamily.org/index.php?title=Main_Page
.. _GNU Scientific Library: https://www.gnu.org/software/gsl/
.. _CUDA: https://developer.nvidia.com/cuda-downloads

The easiest way to build Blackchirp is to open the ``blackchirp.pro`` file in the `Qt Creator`_ IDE, which allows you to easily configure a qmake/C++ kit and control whether the build includes debugging symbols.
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

.. warning::
   The CUDA module has not been tested with recent versions of CUDA or recent graphics cards, and Blackchirp has changed significantly since the last time this module has been used. This module may not compile, and if it does, there may be serious bugs or crashes. Pull requests are welcome!

- ``CONFIG += lif`` enables simultaneous acquitision of CP-FTMW data and Laser-induced fluorescence data (or an equivalent laser scanning experiment with time-gated integration detection). See the `LIF Module <lif.html>`_ page for further details.

Leaving these lines out (or commenting them out) will remove the associated code from the build.
First-time users are advised to leave them out.

Hardware Implementations
------------------------

In Blackchirp, each piece of hardware is made up of two components: the base type of hardware that describes its function within the instrument, and an implementation that corresponds to a specific manufacturer/model.
The ``config.pri`` file allows you to select which specific implementation(s) of hardware you have, as well as which pieces of hardware your instrument has (or those you wish Blackchirp to communicate with and/or control).
For each hardware type, the associated implementations are chosen by listing the model number(s) as shown in the ``config.pri.template`` file.
For example, if your FTMW digitizer is a Tektronix DSA71604C, you would set ``FTMWSCOPE=dsa71604c`` (case insensitive).

Most types of hardware support multiple devices at once.
For example, if you have two pulse generators (a Quantum composers 9518 and a 9528), you can specify ``PGEN=qc9518 9528`` in your ``config.pri`` file.
In that case, the 9518 would appear in Blackchirp as "PulseGenerator.0" and the 9528 as "PulseGenerator.1".
At present, the FtmwScope, AWG, GpibController, LifLaser, and LifScope only support 1 device at a time; all other hardware items may have multiple physical devices enabled simultaneously.

All hardware items have a "virtual" implementation that can serve as a placeholder for a physical device.
While this is primarily intended for development purposes, it can also be useful in some applications.
For example, if you do not have a flow controller setup, you can compile Blackchirp with a virtual flow controller (``FC=virtual``) and enter the names of the sample gases you are using.
Blackchirp will record these values to disk during each experiment.

The only items of hardware that are required for Blackchirp to run are a FTMW digitizer (``FTMWSCOPE``) and at least one "Clock" (``CLOCKS``), which Blackchirp uses to record your specific upconversion/downconversion chain if necessary.
For the clock, the "virtual" implementation (``fixed``) is called a ``FixedClock``, and it contains 6 virtual "outputs" that can be assigned different frequencies.
More information about clocks and setting up your RF configuration can be found in the User Guide (`Hardware Menu - Rf Configuration <user_guide/hardware_menu.html#rf-configuration>`_).


Custom Library/Include Paths
----------------------------

To compile Blackchirp, you must provide the appropriate libraries: qwt, gsl, m, gslcblas, and any other libraries needed for your hardware implementations.
In ``config.pri.template``, an example is provided for Linux assuming these libraries exist in your ``LD_LIBRARY_PATH`` (usually ``/usr/lib``, ``/usr/lib64``, and ``/usr/local/lib``).
If your libraries are located elsewhere, you will need to add them.
Using Qt Creator, you can right-click inside the ``config.pri`` file and use the "Add Library" option, or you can consult the `qmake documentation`_ for details.

.. _qmake documentation: https://doc.qt.io/qt-5/qmake-variable-reference.html#libs

Finally, you will need to ensure that the compiler can find the header files for qwt, eigen3, and gsl.
Blackchirp's source code assumes that include files for each of these libraries is found within a subfolder of your ``INCLUDE`` path: ``qwt6``, ``eigen3``, and ``gsl`` respectively.
If this is not already the case on your system, you can create symbolic links and modify the qmake ``INCLUDEPATH`` variable as needed.
