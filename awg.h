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
#include "awg71002a.h"
class Awg71002a;
typedef Awg71002a AwgHardware;
#else
#include "virtualawg.h"
class VirtualAwg;
typedef VirtualAwg AwgHardware;
#endif
#endif

#endif // AWG_H
