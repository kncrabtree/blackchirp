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

gpu-cuda {
DEFINES += BC_CUDA

# Cuda sources
CUDA_SOURCES += gpuaverager.cu

# Path to cuda toolkit install
CUDA_DIR      = /usr/local/cuda-6.5
INCLUDEPATH  += $$CUDA_DIR/include -isystem /usr/include/qt5 -isystem /usr/include/qt5/QtCore
QMAKE_LIBDIR += $$CUDA_DIR/lib64
CUDA_INC = $$join(INCLUDEPATH,' -I','-I',' ')
CUDA_INT
LIBS += -L$$CUDA_DIR/lib64 -lcuda -lcudart
# GPU architecture
CUDA_ARCH     = sm_50
NVCCFLAGS     = --compiler-options -use_fast_math --ptxas-options=-v -Xcompiler -fPIE -Xcompiler -Wno-attributes

CONFIG(debug, debug|release) {
cuda_d.commands = $$CUDA_DIR/bin/nvcc -D_DEBUG -std=c++11 -m64 -O3 -arch=$$CUDA_ARCH -c $$NVCCFLAGS \
			 $$CUDA_INC ${QMAKE_FILE_NAME} -o ${QMAKE_FILE_OUT} \
			 2>&1 | sed -r \"s/\\(([0-9]+)\\)/:\\1/g\" 1>&2

cuda_d.dependency_type = TYPE_C
cuda_d.depend_command = $$CUDA_DIR/bin/nvcc -O3 -M $$CUDA_INC $$NVCCFLAGS   ${QMAKE_FILE_NAME}

cuda_d.input = CUDA_SOURCES
cuda_d.output = ${OBJECTS_DIR}${QMAKE_FILE_BASE}_cuda.o
QMAKE_EXTRA_COMPILERS += cuda_d
}
else {
cuda.commands = $$CUDA_DIR/bin/nvcc -std=c++11 -m64 -O3 -arch=$$CUDA_ARCH -c $$NVCCFLAGS \
			 $$CUDA_INC $$LIBS ${QMAKE_FILE_NAME} -o ${QMAKE_FILE_OUT} \
			 2>&1 | sed -r \"s/\\(([0-9]+)\\)/:\\1/g\" 1>&2

cuda.dependency_type = TYPE_C
cuda.depend_command = $$CUDA_DIR/bin/nvcc -O3 -M $$CUDA_INC $$NVCCFLAGS   ${QMAKE_FILE_NAME}

cuda.input = CUDA_SOURCES
cuda.output = ${OBJECTS_DIR}${QMAKE_FILE_BASE}_cuda.o
QMAKE_EXTRA_COMPILERS += cuda
}
}

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

QMAKE_CXXFLAGS_RELEASE -= -O2
QMAKE_CXXFLAGS_RELEASE += -O3
QMAKE_CXXFLAGS += -std=c++11


