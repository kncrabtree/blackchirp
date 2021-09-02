#ifndef TEMPERATURECONTROLLER_H
#define TEMPERATURECONTROLLER_H
#include <hardware/core/hardwareobject.h>

#include <hardware/optional/tempcontroller/temperaturecontrollerconfig.h>

class QTimer;

namespace BC::Key::TC {
static const QString key{"TemperatureController"};
static const QString interval{"pollIntervalMs"};
static const QString numChannels{"numChannels"};
static const QString channels{"channels"};
static const QString units{"units"};
static const QString chName{"name"};
static const QString enabled{"enabled"};
static const QString decimals{"decimal"};
}

namespace BC::Aux::TC {
static const QString temperature("Temperature%1");
}

class TemperatureController : public HardwareObject
{
    Q_OBJECT
public:
    explicit TemperatureController(const QString subKey, const QString name, CommunicationProtocol::CommType commType, int channels, QObject *parent =nullptr, bool threaded = false, bool critical = false);
    virtual ~TemperatureController();

    int numChannels() const { return d_numChannels; }

signals:
    void channelEnableUpdate(int,bool,QPrivateSignal);
    void temperatureUpdate(int, double, QPrivateSignal);

public slots:
    void readAll();
    void setChannelEnabled(int ch, bool en);
    void setChannelName(int ch, const QString name);
    double readTemperature(const int ch);
    bool readChannelEnabled(const int ch);
    TemperatureControllerConfig getConfig() const { return d_config; }


    // HardwareObject interface
protected:
    bool prepareForExperiment(Experiment &e) override final;
    virtual AuxDataStorage::AuxDataMap readAuxData() override;
    void initialize() override final;
    bool testConnection() override final;
    void readSettings() override final;

    virtual void tcInitialize() =0;
    virtual bool tcTestConnection() =0;
    virtual double readHwTemperature(const int ch) =0;
    virtual bool readHwChannelEnabled(const int ch) { return d_config.channelEnabled(ch); }
    virtual void setHwChannelEnabled(const int ch, bool en) { d_config.setEnabled(ch,en); }
    virtual void poll();

private:
    const int d_numChannels;
    TemperatureControllerConfig d_config;
    QTimer *p_readTimer;

#if BC_TEMPCONTROLLER == 0
    friend class VirtualTemperatureController;
#endif


    // HardwareObject interface
public:
    QStringList validationKeys() const override;

    // HardwareObject interface
public slots:
    QStringList forbiddenKeys() const override;
};

#endif // TEMPERATURECONTROLLER_H

#ifdef BC_TEMPCONTROLLER
#if BC_TEMPCONTROLLER == 0
#include "virtualtempcontroller.h"
class VirtualTemperatureController;
typedef VirtualTemperatureController TemperatureControllerHardware;
#elif BC_TEMPCONTROLLER == 1
#include "lakeshore218.h"
class Lakeshore218;
typedef Lakeshore218 TemperatureControllerHardware;
#endif
#endif



