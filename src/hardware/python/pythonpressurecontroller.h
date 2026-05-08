#ifndef PYTHONPRESSURECONTROLLER_H
#define PYTHONPRESSURECONTROLLER_H

#include <hardware/optional/pressurecontroller/pressurecontroller.h>
#include <hardware/core/hardwareregistry.h>

#include "pythonhardwarebase.h"

/*!
 * \brief PressureController subclass that dispatches all virtual methods to a Python subprocess via IPC
 *
 * PythonPressureController launches a Python subprocess (via PythonProcess) that
 * loads a user-written pressure controller driver script. All hardware virtual
 * methods are translated to JSON requests sent over stdin/stdout pipes.
 *
 * Communication with actual hardware is relayed: when the Python script
 * calls self.comm.query(), the request is sent back to C++ which performs
 * the operation on p_comm and returns the result.
 *
 * The PressureController base class handles the poll timer, readAuxData(), and
 * prepareForExperiment(). PythonPressureController only needs to implement
 * pcInitialize(), pcTestConnection(), and the nine hw* pure virtuals.
 *
 * The readOnly constructor parameter controls whether setpoint/valve methods
 * are exposed in the GUI. It is read from QSettings before construction and
 * passed to the PressureController base class.
 */
class PythonPressureController : public PressureController, public PythonHardwareBase
{
    Q_OBJECT
public:
    explicit PythonPressureController(const QString &label, QObject *parent = nullptr);

protected:
    void pcInitialize() override;
    bool pcTestConnection() override;

    void pcReadSettings() override;
    void sleep(bool b) override;

private:
    double hwReadPressure() override;
    double hwSetPressureSetpoint(const double val) override;
    double hwReadPressureSetpoint() override;
    void hwSetPressureControlMode(bool enabled) override;
    int hwReadPressureControlMode() override;
    void hwOpenGateValve() override;
    void hwCloseGateValve() override;

};

#endif // PYTHONPRESSURECONTROLLER_H
