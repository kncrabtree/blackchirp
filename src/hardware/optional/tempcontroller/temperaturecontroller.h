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
    explicit TemperatureController(const QString subKey, const QString name, CommunicationProtocol::CommType commType, uint channels, QObject *parent =nullptr, bool threaded = false, bool critical = false);
    virtual ~TemperatureController();

    uint numChannels() const { return d_numChannels; }

signals:
    void channelEnableUpdate(uint,bool,QPrivateSignal);
    void temperatureUpdate(uint, double, QPrivateSignal);

public slots:
    void readAll();
    void setChannelEnabled(uint ch, bool en);
    void setChannelName(uint ch, const QString name);
    double readTemperature(const uint ch);
    bool readChannelEnabled(const uint ch);
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
    virtual double readHwTemperature(const uint ch) =0;
    virtual bool readHwChannelEnabled(const uint ch) { return d_config.channelEnabled(ch); }
    virtual void setHwChannelEnabled(const uint ch, bool en) { d_config.setEnabled(ch,en); }
    virtual void poll();

private:
    const uint d_numChannels;
    TemperatureControllerConfig d_config;
    QTimer *p_readTimer;

    inline static int d_count = 0;

    friend class VirtualTemperatureController;



    // HardwareObject interface
public:
    QStringList validationKeys() const override;

    // HardwareObject interface
public slots:
    QStringList forbiddenKeys() const override;
};

#endif // TEMPERATURECONTROLLER_H




