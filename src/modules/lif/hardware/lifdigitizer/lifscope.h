#ifndef LIFSCOPE_H
#define LIFSCOPE_H

#include <hardware/core/hardwareobject.h>
#include <modules/lif/hardware/lifdigitizer/lifdigitizerconfig.h>
#include <modules/lif/data/lifconfig.h>

namespace BC::Key::LifDigi {
static const QString lifScope{"LifDigitizer"};
}

class LifScope : public HardwareObject, protected LifDigitizerConfig
{
    Q_OBJECT
public:
    LifScope(const QString subKey, const QString name, CommunicationProtocol::CommType commType, QObject *parent = nullptr, bool threaded=true,bool critical=true);
    virtual ~LifScope();

signals:
    void waveformRead(QVector<qint8>);
    void configAcqComplete(LifDigitizerConfig,QPrivateSignal);

public slots:
    virtual void startConfigurationAcquisition(const LifDigitizerConfig &c);

    virtual void readWaveform() =0;
    virtual bool configure(const LifDigitizerConfig &c) =0;

};


#if BC_LIFSCOPE == 1
#include "m4i2211x8.h"
class M4i2211x8;
using LifScopeHardware = M4i2211x8;
#else
#include "virtuallifscope.h"
class VirtualLifScope;
using LifScopeHardware = VirtualLifScope;
#endif

#endif // LIFSCOPE_H
