#ifndef LAKESHORE218_H
#define LAKESHORE218_H
#include <hardware/optional/tempcontroller/temperaturecontroller.h>

class Lakeshore218 : public TemperatureController
{
    Q_OBJECT
public:
    explicit Lakeshore218(const QString& label, QObject* parent = nullptr);

    // HardwareObject interface
public slots:
    QStringList channelNames();
    // TemperatureController interface
protected:
    bool tcTestConnection() override;
    void tcInitialize() override;

    double readHwTemperature(const uint ch) override;

private:
    QTimer *p_readTimer;



};

#endif // LAKESHORE218_H
