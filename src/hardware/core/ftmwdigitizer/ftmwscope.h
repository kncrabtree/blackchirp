#ifndef FTMWSCOPE_H
#define FTMWSCOPE_H

#include <hardware/core/hardwareobject.h>

#include <QByteArray>

#include <data/experiment/ftmwconfig.h>
#include <data/experiment/ftmwdigitizerconfig.h>


namespace BC::Key::FtmwScope {
static const QString ftmwScope("ftmwscope");
static const QString bandwidth("bandwidthMHz");
}

class FtmwScope : public HardwareObject, protected FtmwDigitizerConfig
{
    Q_OBJECT
public:
    explicit FtmwScope(const QString subKey, const QString name, CommunicationProtocol::CommType commType, QObject *parent = nullptr, bool threaded = true, bool critical = true);
    virtual ~FtmwScope();

signals:
    void shotAcquired(const QByteArray data);

public slots:
    virtual void readWaveform() =0;

};

#ifdef BC_FTMWSCOPE
#if BC_FTMWSCOPE == 1
#include "dsa71604c.h"
class Dsa71604c;
typedef Dsa71604c FtmwScopeHardware;
#elif BC_FTMWSCOPE == 2
#include "mso72004c.h"
class MSO72004C;
typedef MSO72004C FtmwScopeHardware;
#elif BC_FTMWSCOPE == 3
#include "m4i2220x8.h"
class M4i2220x8;
typedef M4i2220x8 FtmwScopeHardware;
#elif BC_FTMWSCOPE == 4
#include "dsox92004a.h"
class DSOx92004A;
typedef DSOx92004A FtmwScopeHardware;
#else
#include "virtualftmwscope.h"
class VirtualFtmwScope;
typedef VirtualFtmwScope FtmwScopeHardware;
#endif
#endif

#endif // FTMWSCOPE_H
