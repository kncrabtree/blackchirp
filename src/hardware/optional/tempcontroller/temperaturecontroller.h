#ifndef TEMPERATURECONTROLLER_H
#define TEMPERATURECONTROLLER_H
#include <hardware/core/hardwareobject.h>

class QTimer;

namespace BC::Key::TC {
static const QString key("TemperatureController");
static const QString interval("intervalMs");
static const QString channels("channels");
static const QString units("units");
static const QString name("name");
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
    void temperatureListUpdate(QList<double>, QPrivateSignal);
    void temperatureUpdate(int, double, QPrivateSignal);

public slots:
    QList<double> readAll();
    double readTemperature(const int ch);


    // HardwareObject interface
protected:
    bool prepareForExperiment(Experiment &e) override final;
    virtual AuxDataStorage::AuxDataMap readAuxData() override;
    void initialize() override final;
    bool testConnection() override final;

    virtual void tcInitialize() =0;
    virtual bool tcTestConnection() =0;
    virtual QList<double> readHWTemperatures() = 0;
    virtual double readHwTemperature(const int ch) =0;
    virtual void poll();

private:
    const int d_numChannels;
    QList<double> d_temperatureList;
    QTimer *p_readTimer;

#if BC_TEMPCONTROLLER == 0
    friend class VirtualTemperatureController;
#endif

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



