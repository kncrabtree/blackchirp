#ifndef VALON5009_H
#define VALON5009_H

#include <hardware/core/clock/clock.h>

namespace BC {
namespace Key {
static const QString valon5009("valon5009");
static const QString valon5009Name("Valon Synthesizer 5009");
}
}

class Valon5009 : public Clock
{
public:
    explicit Valon5009(int clockNum, QObject *parent = nullptr);

    // Clock interface
public:
    QStringList channelNames() override { return {"Source 1","Source 2"}; }

protected:
    bool testConnection() override;
    void initializeClock() override;
    bool setHwFrequency(double freqMHz, int outputIndex) override;
    double readHwFrequency(int outputIndex) override;
    bool prepareClock(Experiment &exp) override;

private:
    bool valonWriteCmd(QString cmd);
    QByteArray valonQueryCmd(QString cmd);
};

#endif // VALON5009_H
