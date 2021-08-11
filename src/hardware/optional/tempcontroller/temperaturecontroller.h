#ifndef TEMPERATURECONTROLLER_H
#define TEMPERATURECONTROLLER_H
#include <hardware/core/hardwareobject.h>

#include <hardware/optional/tempcontroller/temperaturecontrollerconfig.h>

class QTimer;

namespace BC::Key::TC {
static const QString key("TemperatureController");
static const QString interval("intervalMs");
static const QString channels("channels");
static const QString units("units");
static const QString chName("name");
static const QString enabled("enabled");
static const QString decimals("decimal");
}

namespace BC::Aux::TC {
static const QString temperature("temperature%1");
}

class TemperatureController : public HardwareObject
{
    Q_OBJECT
public:
    explicit TemperatureController(const QString subKey, const QString name, CommunicationProtocol::CommType commType, int channels, QObject *parent =nullptr, bool threaded = false, bool critical = false);
    virtual ~TemperatureController();

    int numChannels() const { return d_numChannels; }

signals:
    void temperatureUpdate(int, double, QPrivateSignal);

public slots:
    void readAll();
    double readTemperature(const int ch);


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
};

#endif // TEMPERATURECONTROLLER_H


#if BC_TEMPCONTROLLER == 1
#include "lakeshore218.h"
class Lakeshore218;
typedef Lakeshore218 TemperatureControllerHardware;
#else
#include "virtualtempcontroller.h"
class VirtualTemperatureController;
typedef VirtualTemperatureController TemperatureControllerHardware;
#endif



