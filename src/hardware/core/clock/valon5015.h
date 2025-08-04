#ifndef VALON5015_H
#define VALON5015_H

#include <hardware/core/clock/clock.h>

class Valon5015 : public Clock
{
    Q_OBJECT
public:
    explicit Valon5015(const QString& label, QObject* parent = nullptr);

    // HardwareObject interface
public slots:
    QStringList channelNames() override { return {"Source 1"}; }


    // Clock interface
protected:
    bool testClockConnection() override;
    void initializeClock() override;
    bool setHwFrequency(double freqMHz, int outputIndex) override;
    double readHwFrequency(int outputIndex) override;
    bool prepareClock(Experiment &exp) override;

private:
    bool valonWriteCmd(QString cmd);
    QByteArray valonQueryCmd(QString cmd);};

#endif // VALON5015_H
