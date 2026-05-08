#ifndef TEMPERATURECONTROLLER_H
#define TEMPERATURECONTROLLER_H
#include <hardware/core/hardwareobject.h>

#include <data/experiment/hardware/optional/tempcontroller/temperaturecontrollerconfig.h>

class QTimer;


namespace BC::Aux::TC {
inline const QString temperature{"Temperature%1"};
}

class TemperatureController : public HardwareObject
{
    Q_OBJECT
public:
    explicit TemperatureController(const QString& impl, const QString& label, uint channels, QObject *parent = nullptr);
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
    void storeOptHwConfig(Experiment *exp) override { exp->addOptHwConfig(getConfig()); }
    void prepareForShutdown() override;


    // HardwareObject interface
protected:
    bool prepareForExperiment(Experiment &e) override final;
    virtual AuxDataStorage::AuxDataMap readAuxData() override;
    void initialize() override final;
    bool testConnection() override final;
    void hwReadSettings() override final;

    virtual void tcInitialize() =0;
    virtual bool tcTestConnection() =0;
    /*!
     * \brief Driver hook called after the base TemperatureController has
     * refreshed its poll interval. Default is a no-op.
     */
    virtual void tcReadSettings() {}
    virtual double readHwTemperature(const uint ch) =0;
    virtual bool readHwChannelEnabled(const uint ch) { return d_config.channelEnabled(ch); }
    virtual void setHwChannelEnabled(const uint ch, bool en) { d_config.setEnabled(ch,en); }
    virtual void poll();

private:
    const uint d_numChannels;
    TemperatureControllerConfig d_config;
    QTimer *p_readTimer;

    friend class VirtualTemperatureController;



    // HardwareObject interface
public:
    QStringList validationKeys() const override;

};

#endif // TEMPERATURECONTROLLER_H




