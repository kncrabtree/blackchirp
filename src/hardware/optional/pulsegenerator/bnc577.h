#ifndef BNC577_H
#define BNC577_H

#include "qcpulsegenerator.h"

namespace BC::Key::PGen {
static const QString bnc577_4{"bnc577_4"};
static const QString bnc577_8{"bnc577_8"};
static const QString bnc577Name{"Pulse Generator BNC 577"};
}

class Bnc577_4 : public QCPulseGenerator
{
    Q_OBJECT
public:
    explicit Bnc577_4(QObject *parent = nullptr);

    // PulseGenerator interface
protected:
    void initializePGen() override;

    // QCPulseGenerator interface
protected:
    bool pGenWriteCmd(QString cmd) override;
    QByteArray pGenQueryCmd(QString cmd) override;
    QString idResponse() override { return "BNC,577"; }
    QString sysStr() override { return "PULSE0"; }
    QString clock10MHzStr() override { return "EXT10"; }
    QString trigModeBase() override { return ":PULSE:TRIG:MODE"; }
    QString trigEdgeBase() override {return ":PULSE:TRIG:EDGE"; }
};

class Bnc577_8 : public QCPulseGenerator
{
    Q_OBJECT
public:
    explicit Bnc577_8(QObject *parent = nullptr);

    // PulseGenerator interface
protected:
    void initializePGen() override;

    // QCPulseGenerator interface
protected:
    bool pGenWriteCmd(QString cmd) override;
    QByteArray pGenQueryCmd(QString cmd) override;
    QString idResponse() override { return "BNC,577"; }
    QString sysStr() override { return "PULSE0"; }
    QString clock10MHzStr() override { return "EXT10"; }
    QString trigModeBase() override { return ":PULSE:TRIG:MODE"; }
    QString trigEdgeBase() override {return ":PULSE:TRIG:EDGE"; }
};

#endif // BNC577_H
