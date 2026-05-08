#ifndef PYTHONFLOWCONTROLLER_H
#define PYTHONFLOWCONTROLLER_H

#include <hardware/optional/flowcontroller/flowcontroller.h>

#include "pythonhardwarebase.h"

/*!
 * \brief FlowController subclass that dispatches all virtual methods to a Python subprocess via IPC
 *
 * PythonFlowController launches a Python subprocess (via PythonProcess) that
 * loads a user-written flow controller driver script. All hardware virtual
 * methods are translated to JSON requests sent over stdin/stdout pipes.
 *
 * Communication with actual hardware is relayed: when the Python script
 * calls self.comm.query(), the request is sent back to C++ which performs
 * the operation on p_comm and returns the result.
 *
 * The FlowController base class handles the poll timer, readAll(), and
 * prepareForExperiment(). PythonFlowController only needs to implement
 * fcInitialize(), fcTestConnection(), and the eight hw* pure virtuals.
 */
class PythonFlowController : public FlowController, public PythonHardwareBase
{
    Q_OBJECT
public:
    explicit PythonFlowController(const QString &label, QObject *parent = nullptr);

protected:
    void fcInitialize() override;
    bool fcTestConnection() override;

    void fcReadSettings() override;

private:
    void hwSetPressureControlMode(bool enabled) override;
    void hwSetFlowSetpoint(const int ch, const double val) override;
    void hwSetPressureSetpoint(const double val) override;
    double hwReadFlowSetpoint(const int ch) override;
    double hwReadPressureSetpoint() override;
    double hwReadFlow(const int ch) override;
    double hwReadPressure() override;
    int hwReadPressureControlMode() override;
};

#endif // PYTHONFLOWCONTROLLER_H
