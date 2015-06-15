#ifndef AWG_H
#define AWG_H

#include "hardwareobject.h"

class AWG : public HardwareObject
{
    Q_OBJECT
public:
    AWG(QObject *parent);
    ~AWG();
};


#ifdef BC_AWG
#if BC_AWG==1
#include "awg70002a.h"
class AWG70002a;
typedef AWG70002a AwgHardware;
#else
#include "virtualawg.h"
class VirtualAwg;
typedef VirtualAwg AwgHardware;
#endif
#endif

#endif // AWG_H
