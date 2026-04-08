#ifndef PYTHONGPIBCONTROLLER_H
#define PYTHONGPIBCONTROLLER_H

#include <hardware/optional/gpibcontroller/gpibcontroller.h>
#include <hardware/core/hardwareregistry.h>

#include "pythonhardwarebase.h"

/*!
 * \brief GpibController subclass that dispatches all virtual methods to a Python subprocess via IPC
 *
 * PythonGpibController launches a Python subprocess (via PythonProcess) that
 * loads a user-written GPIB controller driver script. All hardware virtual
 * methods are translated to JSON requests sent over stdin/stdout pipes.
 *
 * The GpibController base class handles bus arbitration (writeCmd, writeBinary,
 * queryCmd) including mutex locking and address management. PythonGpibController
 * only needs to implement initialize(), testConnection(), readAddress(), and
 * setAddress() -- the four methods that interact directly with hardware.
 *
 * Communication with actual hardware is relayed: when the Python script
 * calls self.comm.query(), the request is sent back to C++ which performs
 * the operation on p_comm and returns the result.
 */
class PythonGpibController : public GpibController, public PythonHardwareBase
{
    Q_OBJECT
public:
    explicit PythonGpibController(const QString &label, QObject *parent = nullptr);

    static QVector<HwConfigParam> configParams();

protected:
    void initialize() override;
    bool testConnection() override;

    bool readAddress() override;
    bool setAddress(int a) override;

    void sleep(bool b) override;
    void readSettings() override;
    QStringList forbiddenKeys() const override;
};

#endif // PYTHONGPIBCONTROLLER_H
