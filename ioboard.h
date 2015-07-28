#ifndef IOBOARD_H
#define IOBOARD_H

#include "hardwareobject.h"

#include "ioboardconfig.h"

class IOBoard : public HardwareObject
{
    Q_OBJECT
public:
    explicit IOBoard(QObject *parent = nullptr);
    virtual ~IOBoard();

protected:
    IOBoardConfig d_config;
    int d_numAnalog;
    int d_numDigital;
    int d_reservedAnalog;
    int d_reservedDigital;

    void readSettings();

};

#ifdef BC_IOBOARD
#if BC_IOBOARD == 1
#include "labjacku3.h"
class LabjackU3;
typedef LabjackU3 IOBoardHardware;
#else
#include "virtualioboard.h"
class VirtualIOBoard;
typedef VirtualIOBoard IOBoardHardware;
#endif
#endif

#endif // IOBOARD_H
