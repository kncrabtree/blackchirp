#ifndef PYTHONCLOCK_H
#define PYTHONCLOCK_H

#include <hardware/core/clock/clock.h>
#include <hardware/core/hardwareregistry.h>

#include "pythonhardwarebase.h"

namespace BC::Key::PythonClock {
static const QString numOutputs{"numOutputs"};
}

/*!
 * \brief Clock subclass that dispatches all virtual methods to a Python subprocess via IPC
 *
 * PythonClock launches a Python subprocess (via PythonProcess) that loads a
 * user-written clock driver script. All hardware virtual methods are translated
 * to JSON requests sent over stdin/stdout pipes.
 *
 * Communication with actual hardware is relayed: when the Python script calls
 * self.comm.query(), the request is sent back to C++ which performs the
 * operation on p_comm and returns the result.
 *
 * The Clock base class handles role assignment, multiplier factors,
 * prepareForExperiment(), and readAll(). PythonClock only needs to implement
 * initializeClock(), testClockConnection(), setHwFrequency(), and
 * readHwFrequency().
 *
 * Because Clock requires numOutputs and tunable at construction time (before
 * SettingsStorage exists), PythonClock reads those values directly from
 * QSettings in the constructor member-initializer list.
 */
class PythonClock : public Clock, public PythonHardwareBase
{
    Q_OBJECT
public:
    explicit PythonClock(const QString &label, QObject *parent = nullptr);

protected:
    void initializeClock() override;
    bool testClockConnection() override;
    bool setHwFrequency(double freqMHz, int outputIndex = 0) override;
    double readHwFrequency(int outputIndex = 0) override;

    void readSettings() override;
    void sleep(bool b) override;
    QStringList forbiddenKeys() const override;

};

#endif // PYTHONCLOCK_H
