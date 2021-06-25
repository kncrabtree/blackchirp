#ifndef GPIBCONTROLLER_H
#define GPIBCONTROLLER_H

#include <hardware/core/hardwareobject.h>

namespace BC::Key {
static const QString gpibController("gpibController");
}

class GpibController : public HardwareObject
{
	Q_OBJECT
public:
    GpibController(const QString subKey, const QString name, CommunicationProtocol::CommType commType, QObject *parent = nullptr, bool threaded = true, bool critical = false);
    virtual ~GpibController();

	bool writeCmd(int address, QString cmd);
    bool writeBinary(int address, QByteArray dat);
    QByteArray queryCmd(int address, QString cmd, bool suppressError=false);
    virtual QString queryTerminator() const { return QString(); }

protected:
    virtual bool readAddress() =0;
    virtual bool setAddress(int a) =0;

	int d_currentAddress;
};


#if BC_GPIBCONTROLLER == 1
#include "prologixgpiblan.h"
class PrologixGpibLan;
typedef PrologixGpibLan GpibControllerHardware;
#else
#include "virtualgpibcontroller.h"
class VirtualGpibController;
typedef VirtualGpibController GpibControllerHardware;
#endif

#endif // GPIBCONTROLLER_H
