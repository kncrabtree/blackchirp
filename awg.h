#ifndef AWG_H
#define AWG_H

#include "hardwareobject.h"

class AWG : public HardwareObject
{
    Q_OBJECT
public:
    AWG(QObject *parent);
    virtual ~AWG();
};


#ifdef BC_AWG
#if BC_AWG==1
#include "awg70002a.h"
class AWG70002a;
typedef AWG70002a AwgHardware;
#elif BC_AWG==2
#include "awg7122b.h"
class AWG7122B;
typedef AWG7122B AwgHardware;
#elif BC_AWG==3
#include "ad9914.h"
class AD9914;
typedef AD9914 AwgHardware;
#else
#include "virtualawg.h"
class VirtualAwg;
typedef VirtualAwg AwgHardware;
#endif
#endif

#endif // AWG_H
