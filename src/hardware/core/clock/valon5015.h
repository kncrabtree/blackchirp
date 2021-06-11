#ifndef VALON5015_H
#define VALON5015_H

#include <src/hardware/core/clock/clock.h>

namespace BC {
namespace Key {
static const QString valon5015("valon5015");
static const QString valon5015Name("Valon Synthesizer 5015");
}
}

class Valon5015 : public Clock
{
    Q_OBJECT
public:
    explicit Valon5015(int clockNum, QObject* parent = nullptr);

    // HardwareObject interface
public slots:
    void readSettings() override;
    QStringList channelNames() override;


    // Clock interface
protected:
    bool testConnection() override;
    void initializeClock() override;
    bool setHwFrequency(double freqMHz, int outputIndex) override;
    double readHwFrequency(int outputIndex) override;
    bool prepareClock(Experiment &exp) override;

private:
    bool valonWriteCmd(QString cmd);
    QByteArray valonQueryCmd(QString cmd);

    bool d_lockToExt10MHz;
};

#endif // VALON5015_H
