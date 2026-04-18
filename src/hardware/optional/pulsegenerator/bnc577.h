#ifndef BNC577_H
#define BNC577_H

#include "qcpulsegenerator.h"

namespace BC::Key::PGen {
inline constexpr QLatin1StringView bnc577{"bnc577"};
inline constexpr QLatin1StringView bnc577Name{"Pulse Generator BNC 577"};
}

class Bnc577 : public QCPulseGenerator
{
    Q_OBJECT
public:
    explicit Bnc577(const QString& label, QObject *parent = nullptr);

    // PulseGenerator interface
protected:
    void initializePGen() override;

    // QCPulseGenerator interface
protected:
    bool pGenWriteCmd(const QString &cmd) override;
    QByteArray pGenQueryCmd(const QString &cmd) override;
    QString idResponse() override { return "BNC,577"; }
    QString sysStr() override { return "PULSE0"; }
    QString clock10MHzStr() override { return "EXT10"; }
    QString trigModeBase() override { return ":PULSE:TRIG:MODE"; }
    QString trigEdgeBase() override { return ":PULSE:TRIG:EDGE"; }
};

#endif // BNC577_H
