#ifndef QC9518_H
#define QC9518_H

#include <hardware/optional/pulsegenerator/qcpulsegenerator.h>

namespace BC::Key {
static const QString qc9518{"QC9518"};
static const QString qc9518Name("Pulse Generator QC 9518");
}

class Qc9518 : public QCPulseGenerator
{
public:
    explicit Qc9518(QObject *parent = nullptr);
    ~Qc9518();

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
    const QString id{"9518+"};
    const QString sys{"SPULSE"};
    const QString clock{"1"};
    const QString tb{":SPULSE:EXT:MOD"};
};

#endif // QC9518_H
