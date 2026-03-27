#ifndef PYTHONTESTHARDWARE_H
#define PYTHONTESTHARDWARE_H

#ifdef BC_PYTHON_HARDWARE

#include <hardware/core/hardwareobject.h>

#include <memory>

class PythonProcess;

namespace BC::Key::PythonTest {
static const QString pythonScript{"pythonScript"};
static const QString pythonClass{"pythonClass"};
}

/*!
 * \brief HardwareObject subclass that dispatches to a Python subprocess via IPC
 *
 * PythonTestHardware launches a Python subprocess (via PythonProcess) that
 * loads a user-written hardware script. All HardwareObject virtual methods
 * are translated to JSON requests sent over stdin/stdout pipes.
 *
 * Communication with actual hardware is relayed: when the Python script
 * calls self.comm.query(), the request is sent back to C++ which performs
 * the operation on p_comm and returns the result.
 */
class PythonTestHardware : public HardwareObject
{
    Q_OBJECT
public:
    explicit PythonTestHardware(const QString &label, QObject *parent = nullptr);
    ~PythonTestHardware() override;

protected:
    void initialize() override;
    bool testConnection() override;

private:
    AuxDataStorage::AuxDataMap readAuxData() override;
    AuxDataStorage::AuxDataMap readValidationData() override;
    bool prepareForExperiment(Experiment &exp) override;
    void beginAcquisition() override;
    void endAcquisition() override;
    void sleep(bool b) override;
    void readSettings() override;
    QStringList forbiddenKeys() const override;

    AuxDataStorage::AuxDataMap parseAuxDataResult(const QJsonObject &response);
    bool startPythonProcess();
    QString findHostScript() const;

    std::unique_ptr<PythonProcess> pu_process;
};

#endif // BC_PYTHON_HARDWARE
#endif // PYTHONTESTHARDWARE_H
