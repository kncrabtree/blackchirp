#-------------------------------------------------
#
# Project created by QtCreator 2015-02-11T14:07:58
#
#-------------------------------------------------

QT       += core gui network

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets serialport

#Enable CUDA GPU support
CONFIG += gpu-cuda

TARGET = blackchirp
TEMPLATE = app


SOURCES += main.cpp

include(acquisition.pri)
include(gui.pri)
include(data.pri)
include(hardware.pri)


unix:!macx: LIBS += -lqwt -lgsl -lm -lgslcblas

#------------------------------------------------
# The following defines allow running the code without real hardware.
# Uncomment the appropriate lines to simulate hardware.
# -----------------------------------------------

# Simulates ALL hardware
#DEFINES += BC_NOHARDWARE

# Simulates ALL RS232 devices
#DEFINES += BC_NORS232

# Simulates ALL TCP devices
#DEFINES += BC_NOTCP

# Simulates FTMW Oscilloscope (uncomment DEFINES and RESOURCES lines)
DEFINES += BC_NOFTSCOPE
RESOURCES += virtualdata.qrc


gpu-cuda {
DEFINES += BC_CUDA

# Cuda sources
CUDA_SOURCES += gpukernels.cu

# Path to cuda toolkit install
CUDA_DIR      = /usr/local/cuda-6.5
# Path to header and libs files
INCLUDEPATH  += $$CUDA_DIR/include
QMAKE_LIBDIR += $$CUDA_DIR/lib64     # Note I'm using a 64 bits Operating system
# libs used in your code
LIBS += -lcuda -lcudart
# GPU architecture
CUDA_ARCH     = sm_50                # Yeah! I've a new device. Adjust with your compute capability
# Here are some NVCC flags I've always used by default.
NVCCFLAGS     = --compiler-options -fno-strict-aliasing -use_fast_math --ptxas-options=-v

# Prepare the extra compiler configuration (taken from the nvidia forum - i'm not an expert in this part)
CUDA_INC = $$join(INCLUDEPATH,' -I','-I',' ')

cuda.commands = $$CUDA_DIR/bin/nvcc -m64 -O3 -arch=$$CUDA_ARCH -c $$NVCCFLAGS \
			 $$CUDA_INC $$LIBS  ${QMAKE_FILE_NAME} -o ${QMAKE_FILE_OUT} \
# nvcc error printout format ever so slightly different from gcc
# http://forums.nvidia.com/index.php?showtopic=171651
			# 2>&1 | sed -r 's/\\(([0-9]+)\\)/:\\1/g' 1>&2

cuda.dependency_type = TYPE_C # there was a typo here. Thanks workmate!
cuda.depend_command = $$CUDA_DIR/bin/nvcc -O2 -M $$CUDA_INC $$NVCCFLAGS   ${QMAKE_FILE_NAME}

cuda.input = CUDA_SOURCES
cuda.output = ${OBJECTS_DIR}${QMAKE_FILE_BASE}_cuda.o
# Tell Qt that we want add more stuff to the Makefile
QMAKE_EXTRA_COMPILERS += cuda
}
