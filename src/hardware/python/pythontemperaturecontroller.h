#ifndef PYTHONTEMPERATURECONTROLLER_H
#define PYTHONTEMPERATURECONTROLLER_H

#ifdef BC_PYTHON_HARDWARE

#include <hardware/optional/tempcontroller/temperaturecontroller.h>
#include <hardware/core/hardwareregistry.h>

#include "pythonhardwarebase.h"

/*!
 * \brief TemperatureController subclass that dispatches all virtual methods to a Python subprocess via IPC
 *
 * PythonTemperatureController launches a Python subprocess (via PythonProcess) that
 * loads a user-written temperature controller driver script. All hardware virtual
 * methods are translated to JSON requests sent over stdin/stdout pipes.
 *
 * Communication with actual hardware is relayed: when the Python script
 * calls self.comm.query(), the request is sent back to C++ which performs
 * the operation on p_comm and returns the result.
 *
 * The TemperatureController base class handles the poll timer, readAll(), and
 * prepareForExperiment(). PythonTemperatureController only needs to implement
 * tcInitialize(), tcTestConnection(), and readHwTemperature().
 */
class PythonTemperatureController : public TemperatureController, public PythonHardwareBase
{
    Q_OBJECT
public:
    explicit PythonTemperatureController(const QString &label, QObject *parent = nullptr);

    static QVector<HwConfigParam> configParams();

protected:
    void tcInitialize() override;
    bool tcTestConnection() override;
    double readHwTemperature(const uint ch) override;

    void sleep(bool b) override;
    QStringList forbiddenKeys() const override;

};

#endif // BC_PYTHON_HARDWARE
#endif // PYTHONTEMPERATURECONTROLLER_H
