#ifndef PYTHONIOBOARD_H
#define PYTHONIOBOARD_H

#ifdef BC_PYTHON_HARDWARE

#include <hardware/optional/ioboard/ioboard.h>

#include <memory>

class PythonProcess;

namespace BC::Key::PythonIOBoard {
static const QString pythonScript{"pythonScript"};
static const QString pythonClass{"pythonClass"};
}

/*!
 * \brief IOBoard subclass that dispatches virtual methods to a Python subprocess via IPC
 *
 * PythonIOBoard launches a Python subprocess (via PythonProcess) that loads a
 * user-written IOBoard driver script. The two pure virtual methods required by
 * IOBoard — readAnalogChannels() and readDigitalChannels() — are translated to
 * JSON requests sent over stdin/stdout pipes.
 *
 * The base IOBoard class handles readAuxData() and readValidationData() by
 * delegating to those two virtuals, so only the channel-reading methods need to
 * be dispatched to Python.
 */
class PythonIOBoard : public IOBoard
{
    Q_OBJECT
public:
    explicit PythonIOBoard(const QString &label, QObject *parent = nullptr);
    ~PythonIOBoard() override;

protected:
    void initialize() override;
    bool testConnection() override;

    // IOBoard pure virtuals
    std::map<int, double> readAnalogChannels() override;
    std::map<int, bool> readDigitalChannels() override;

    void sleep(bool b) override;
    void readSettings() override;
    QStringList forbiddenKeys() const override;

private:
    bool startPythonProcess();
    QString findHostScript() const;

    std::unique_ptr<PythonProcess> pu_process;
};

#endif // BC_PYTHON_HARDWARE
#endif // PYTHONIOBOARD_H
