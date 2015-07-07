#ifndef GPIBCONTROLLER_H
#define GPIBCONTROLLER_H

#include "hardwareobject.h"

class GpibController : public HardwareObject
{
	Q_OBJECT
public:
	GpibController(QObject *parent = nullptr);
	~GpibController();

	bool writeCmd(int address, QString cmd);
    bool writeBinary(int address, QByteArray dat);
	QByteArray queryCmd(int address, QString cmd);

protected:
    virtual bool readAddress() =0;
    virtual bool setAddress(int a) =0;

	int d_currentAddress;
};

#ifdef BC_GPIBCONTROLLER
#if BC_GPIBCONTROLLER == 1
#include "prologixgpiblan.h"
class PrologixGpibLan;
typedef PrologixGpibLan GpibControllerHardware;
#else
#include "virtualgpibcontroller.h"
class VirtualGpibController;
typedef VirtualGpibController GpibControllerHardware;
#endif
#endif

#endif // GPIBCONTROLLER_H
