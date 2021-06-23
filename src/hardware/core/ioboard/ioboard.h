#ifndef IOBOARD_H
#define IOBOARD_H

#include <src/hardware/core/hardwareobject.h>

#include <src/data/experiment/ioboardconfig.h>

namespace BC::Key::IOB {
static const QString ioboard("ioboard");
}

class IOBoard : public HardwareObject
{
    Q_OBJECT
public:
    explicit IOBoard(const QString subKey, const QString name, CommunicationProtocol::CommType commType, QObject *parent = nullptr, bool threaded=true,bool critical=false);
    virtual ~IOBoard();

public slots:
    virtual void readSettings() override final;

protected:
    virtual void readIOBSettings() =0;
    IOBoardConfig d_config;
    int d_numAnalog;
    int d_numDigital;
    int d_reservedAnalog;
    int d_reservedDigital;


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
