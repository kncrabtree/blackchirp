#ifndef PYTHONAWG_H
#define PYTHONAWG_H

#ifdef BC_PYTHON_HARDWARE

#include <hardware/optional/chirpsource/awg.h>

#include <memory>

class PythonProcess;

namespace BC::Key::PythonAwg {
static const QString pythonScript{"pythonScript"};
static const QString pythonClass{"pythonClass"};
}

/*!
 * \brief AWG subclass that dispatches all virtual methods to a Python subprocess via IPC
 *
 * PythonAwg launches a Python subprocess (via PythonProcess) that loads a
 * user-written AWG driver script. All HardwareObject virtual methods are
 * translated to JSON requests sent over stdin/stdout pipes.
 *
 * Communication with actual hardware is relayed: when the Python script
 * calls self.comm.query(), the request is sent back to C++ which performs
 * the operation on p_comm and returns the result.
 */
class PythonAwg : public AWG
{
    Q_OBJECT
public:
    explicit PythonAwg(const QString &label, QObject *parent = nullptr);
    ~PythonAwg() override;

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
#endif // PYTHONAWG_H
