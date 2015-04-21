#ifndef AWG_H
#define AWG_H

#include "hardwareobject.h"

#if BC_AWG==1
class Awg71002a;
typedef Awg71002a AwgHardware;
#else
class VirtualAwg;
typedef VirtualAwg AwgHardware;
#endif

class AWG : public HardwareObject
{
    Q_OBJECT
public:
    AWG(QObject *parent);
    ~AWG();

};

#endif // AWG_H
