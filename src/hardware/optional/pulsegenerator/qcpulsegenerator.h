#ifndef QCPULSEGENERATOR_H
#define QCPULSEGENERATOR_H

#include "pulsegenerator.h"

namespace BC::Key::PGen {
#ifdef BC_PGEN_QC9528
static const QString qc9528{"qc9528"};
static const QString qc9528Name("Pulse Generator QC 9528");
#endif
#ifdef BC_PGEN_QC9518
static const QString qc9518{"QC9518"};
static const QString qc9518Name("Pulse Generator QC 9518");
#endif
#ifdef BC_PGEN_QC9214
static const QString qc9214{"QC9214"};
static const QString qc9214Name("Pulse Generator QC 9214");
#endif
}


class QCPulseGenerator : public PulseGenerator
{
    Q_OBJECT
public:
    explicit QCPulseGenerator(const QString subKey, const QString name, CommunicationProtocol::CommType commType, int numChannels, QObject *parent = nullptr, bool threaded = false, bool critical = false);
    virtual ~QCPulseGenerator();



    // HardwareObject interface
protected:
    bool testConnection() override final;

    // PulseGenerator interface
protected:
    bool setChWidth(const int index, const double width) override final;
    bool setChDelay(const int index, const double delay) override final;
    bool setChActiveLevel(const int index, const PulseGenConfig::ActiveLevel level) override final;
    bool setChEnabled(const int index, const bool en) override final;
    bool setChSyncCh(const int index, const int syncCh) override final;
    bool setChMode(const int index, const PulseGenConfig::ChannelMode mode) override final;
    bool setChDutyOn(const int index, const int pulses) override final;
    bool setChDutyOff(const int index, const int pulses) override final;
    bool setHwPulseMode(PulseGenConfig::PGenMode mode) override final;
    bool setHwRepRate(double rr) override final;
    bool setHwPulseEnabled(bool en) override final;
    double readChWidth(const int index) override final;
    double readChDelay(const int index) override final;
    PulseGenConfig::ActiveLevel readChActiveLevel(const int index) override final;
    bool readChEnabled(const int index) override final;
    int readChSynchCh(const int index) override final;
    PulseGenConfig::ChannelMode readChMode(const int index) override final;
    int readChDutyOn(const int index) override final;
    int readChDutyOff(const int index) override final;
    PulseGenConfig::PGenMode readHwPulseMode() override final;
    double readHwRepRate() override final;
    bool readHwPulseEnabled() override final;

protected:
    void lockKeys(bool lock);
    virtual bool pGenWriteCmd(QString cmd) =0;
    virtual QByteArray pGenQueryCmd(QString cmd) =0;


    virtual QString idResponse() =0;
    virtual QString sysStr() =0;
    virtual QString clock10MHzStr() =0;
    virtual QString trigModeBase() =0;
    virtual QString trigEdgeBase() =0;

private:
    const QStringList d_channels{"T0","CHA","CHB","CHC","CHD","CHE","CHF","CHG","CHH"};
};

#ifdef BC_PGEN_QC9518
class Qc9518 : public QCPulseGenerator
{
    Q_OBJECT
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
    inline QString idResponse() override { return id; }
    inline QString sysStr() override { return sys; }
    inline QString clock10MHzStr() override { return clock; }
    inline QString trigModeBase() override { return tb; }
    inline QString trigEdgeBase() override { return te; }

private:
    const QString id{"9518+"};
    const QString sys{"SPULSE"};
    const QString clock{"1"};
    const QString tb{":SPULSE:EXT:MOD"};
    const QString te{":SPULSE:EXT:EDGE"};
};
#endif

#ifdef BC_PGEN_QC9528
class Qc9528 : public QCPulseGenerator
{
    Q_OBJECT
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
    inline QString idResponse() override { return id; }
    inline QString sysStr() override { return sys; }
    inline QString clock10MHzStr() override { return clock; }
    inline QString trigModeBase() override { return tb; }
    inline QString trigEdgeBase() override { return te; }

private:
    const QString id{"QC,9528"};
    const QString sys{"PULSE0"};
    const QString clock{"EXT10"};
    const QString tb{":PULSE0:TRIGGER:MODE"};
    const QString te{":PULSE0:TRIGGER:LOGIC"};
};
#endif

#ifdef BC_PGEN_QC9214
class Qc9214 : public QCPulseGenerator
{
    Q_OBJECT
public:
    explicit Qc9214(QObject *parent = nullptr);
    ~Qc9214();

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
    inline QString idResponse() override { return id; }
    inline QString sysStr() override { return sys; }
    inline QString clock10MHzStr() override { return clock; }
    inline QString trigModeBase() override { return tb; }
    inline QString trigEdgeBase() override { return te; }

private:
    const QString id{"QC,9214"};
    const QString sys{"PULSE0"};
    const QString clock{"EXT10"};
    const QString tb{":PULSE0:EXT:MOD"};
    const QString te{":PULSE0:EXT:EDGE"};
};
#endif

#endif // QCPULSEGENERATOR_H
