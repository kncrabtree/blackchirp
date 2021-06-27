#ifndef LAKESHORE218_H
#define LAKESHORE218_H
#include <hardware/optional/tempcontroller/temperaturecontroller.h>

namespace BC::Key::TC {
static const QString lakeshore218("lakeshore218");
static const QString lakeshore218Name("Lakeshore 218 Temperature Controller");
}

class Lakeshore218 : public TemperatureController
{
    Q_OBJECT
public:
    explicit Lakeshore218(QObject* parent = nullptr);

    // HardwareObject interface
public slots:
    QStringList channelNames();
    // TemperatureController interface
protected:
    bool tcTestConnection() override;
    void tcInitialize() override;

    QList<double> readHWTemperatures() override;
    double readHwTemperature(const int ch) override;

private:
    QTimer *p_readTimer;



};

#endif // LAKESHORE218_H
