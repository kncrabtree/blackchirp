#ifndef PYTHONPULSEGENERATOR_H
#define PYTHONPULSEGENERATOR_H

#ifdef BC_PYTHON_HARDWARE

#include <hardware/optional/pulsegenerator/pulsegenerator.h>
#include <hardware/core/hardwareregistry.h>

#include "pythonhardwarebase.h"

/*!
 * \brief PulseGenerator subclass that dispatches all virtual methods to a Python subprocess via IPC
 *
 * PythonPulseGenerator launches a Python subprocess (via PythonProcess) that
 * loads a user-written pulse generator driver script. All hardware virtual
 * methods are translated to JSON requests sent over stdin/stdout pipes.
 *
 * Communication with actual hardware is relayed: when the Python script
 * calls self.comm.query(), the request is sent back to C++ which performs
 * the operation on p_comm and returns the result.
 *
 * The PulseGenerator base class handles prepareForExperiment() (via setAll()),
 * readAll(), and the public slot interface. PythonPulseGenerator only needs
 * to implement initializePGen(), testConnection(), and the ~22 hw* pure
 * virtuals. sleep() is final in PulseGenerator and calls setHwPulseEnabled(false)
 * internally, so it is handled automatically through IPC.
 *
 * The numChannels constructor parameter is read from QSettings before
 * construction and passed to the PulseGenerator base class.
 */
class PythonPulseGenerator : public PulseGenerator, public PythonHardwareBase
{
    Q_OBJECT
public:
    explicit PythonPulseGenerator(const QString &label, QObject *parent = nullptr);

    static QVector<HwConfigParam> configParams();

protected:
    void initializePGen() override;
    bool testConnection() override;

    void readSettings() override;
    QStringList forbiddenKeys() const override;

private:
    // Per-channel set virtuals
    bool setChWidth(const int index, const double width) override;
    bool setChDelay(const int index, const double delay) override;
    bool setChActiveLevel(const int index, const PulseGenConfig::ActiveLevel level) override;
    bool setChEnabled(const int index, const bool en) override;
    bool setChSyncCh(const int index, const int syncCh) override;
    bool setChMode(const int index, const PulseGenConfig::ChannelMode mode) override;
    bool setChDutyOn(const int index, const int pulses) override;
    bool setChDutyOff(const int index, const int pulses) override;

    // Global set virtuals
    bool setHwPulseMode(PulseGenConfig::PGenMode mode) override;
    bool setHwRepRate(double rr) override;
    bool setHwPulseEnabled(bool en) override;

    // Per-channel read virtuals
    double readChWidth(const int index) override;
    double readChDelay(const int index) override;
    PulseGenConfig::ActiveLevel readChActiveLevel(const int index) override;
    bool readChEnabled(const int index) override;
    int readChSynchCh(const int index) override;
    PulseGenConfig::ChannelMode readChMode(const int index) override;
    int readChDutyOn(const int index) override;
    int readChDutyOff(const int index) override;

    // Global read virtuals
    PulseGenConfig::PGenMode readHwPulseMode() override;
    double readHwRepRate() override;
    bool readHwPulseEnabled() override;
};

#endif // BC_PYTHON_HARDWARE
#endif // PYTHONPULSEGENERATOR_H
