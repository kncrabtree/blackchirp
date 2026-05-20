#ifndef QCPULSEGENERATOR_H
#define QCPULSEGENERATOR_H

#include "pulsegenerator.h"

namespace BC::Key::PGen {
// All QC pulse generator keys always available - no conditional compilation
inline constexpr QLatin1StringView qc9520series{"qc9520series"};
inline const QString qc9520seriesName{"QuantumComposers 9520 Series Pulse Generator"};
inline constexpr QLatin1StringView qc9510series{"qc9510series"};
inline const QString qc9510seriesName{"QuantumComposers 9510 Series Pulse Generator"};
inline constexpr QLatin1StringView qc9210series{"qc9210series"};
inline const QString qc9210seriesName{"QuantumComposers 9210 Series Pulse Generator"};
}


class QCPulseGenerator : public PulseGenerator
{
    Q_OBJECT
public:
    explicit QCPulseGenerator(const QString& impl, const QString& label, int numChannels, QObject *parent = nullptr);
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
    virtual bool pGenWriteCmd(const QString &cmd) =0;
    virtual QByteArray pGenQueryCmd(const QString &cmd) =0;


    virtual QString idResponse() =0;
    virtual QString sysStr() =0;
    virtual QString clock10MHzStr() =0;
    virtual QString trigModeBase() =0;
    virtual QString trigEdgeBase() =0;

private:
    const QStringList d_channels{"T0","CHA","CHB","CHC","CHD","CHE","CHF","CHG","CHH"};
};

// All QC pulse generator implementations always available - no conditional compilation
class Qc9510Series : public QCPulseGenerator
{
    Q_OBJECT
public:
    explicit Qc9510Series(const QString& label, QObject *parent = nullptr);
    ~Qc9510Series();

    // HardwareObject interface
public slots:
    void beginAcquisition() override;
    void endAcquisition() override;

protected:
    void initializePGen() override;


    // QCPulseGenerator interface
protected:
    bool pGenWriteCmd(const QString &cmd) override;
    QByteArray pGenQueryCmd(const QString &cmd) override;
    inline QString idResponse() override { return id; }
    inline QString sysStr() override { return sys; }
    inline QString clock10MHzStr() override { return clock; }
    inline QString trigModeBase() override { return tb; }
    inline QString trigEdgeBase() override { return te; }

private:
    const QString id{"951"};
    const QString sys{"SPULSE"};
    const QString clock{"1"};
    const QString tb{":SPULSE:EXT:MOD"};
    const QString te{":SPULSE:EXT:EDGE"};
};

class Qc9520Series : public QCPulseGenerator
{
    Q_OBJECT
public:
    explicit Qc9520Series(const QString& label, QObject *parent = nullptr);
    ~Qc9520Series();

    // HardwareObject interface
public slots:
    void beginAcquisition() override;
    void endAcquisition() override;


protected:
    void initializePGen() override;


    // QCPulseGenerator interface
protected:
    bool pGenWriteCmd(const QString &cmd) override;
    QByteArray pGenQueryCmd(const QString &cmd) override;
    inline QString idResponse() override { return id; }
    inline QString sysStr() override { return sys; }
    inline QString clock10MHzStr() override { return clock; }
    inline QString trigModeBase() override { return tb; }
    inline QString trigEdgeBase() override { return te; }

private:
    const QString id{"QC,952"};
    const QString sys{"PULSE0"};
    const QString clock{"EXT10"};
    const QString tb{":PULSE:TRIG:MODE"};
    const QString te{":PULS:TRIG:EDGE"};
};

class Qc9210Series : public QCPulseGenerator
{
    Q_OBJECT
public:
    explicit Qc9210Series(const QString& label, QObject *parent = nullptr);
    ~Qc9210Series();

    // HardwareObject interface
public slots:
    void beginAcquisition() override;
    void endAcquisition() override;


protected:
    void initializePGen() override;


    // QCPulseGenerator interface
protected:
    bool pGenWriteCmd(const QString &cmd) override;
    QByteArray pGenQueryCmd(const QString &cmd) override;
    inline QString idResponse() override { return id; }
    inline QString sysStr() override { return sys; }
    inline QString clock10MHzStr() override { return clock; }
    inline QString trigModeBase() override { return tb; }
    inline QString trigEdgeBase() override { return te; }

private:
    const QString id{"QC,921"};
    const QString sys{"PULSE0"};
    const QString clock{"EXT10"};
    const QString tb{":PULSE0:EXT:MOD"};
    const QString te{":PULSE0:EXT:EDGE"};
};

#endif // QCPULSEGENERATOR_H
