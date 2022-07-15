#ifndef QC9528_H
#define QC9528_H

#include <hardware/optional/pulsegenerator/qcpulsegenerator.h>

namespace BC::Key {
static const QString qc9528{"qc9528"};
static const QString qc9528Name("Pulse Generator QC 9528");
}

class Qc9528 : public QCPulseGenerator
{
public:
    explicit Qc9528(QObject *parent = nullptr);
    ~Qc9528();

    // HardwareObject interface
public slots:
    void beginAcquisition() override;
    void endAcquisition() override;


protected:
    void initializePGen() override;


    // QCPulseGenerator interface
protected:
    bool pGenWriteCmd(QString cmd) override;
    QByteArray pGenQueryCmd(QString cmd) override;
    inline QString idResponse() override { return id; };
    inline QString sysStr() override { return sys; };
    inline QString clock10MHzStr() override { return clock; };
    inline QString trigBase() override { return tb; };

private:
    const QString id{"QC,9528"};
    const QString sys{"PULSE0"};
    const QString clock{"EXT10"};
    const QString tb{":PULSE0:TRIG:MOD"};
};

#endif // QC9528_H
