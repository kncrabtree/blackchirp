#ifndef PYTHONIOBOARD_H
#define PYTHONIOBOARD_H

#include <hardware/optional/ioboard/ioboard.h>
#include <hardware/core/hardwareregistry.h>

#include "pythonhardwarebase.h"

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
class PythonIOBoard : public IOBoard, public PythonHardwareBase
{
    Q_OBJECT
public:
    explicit PythonIOBoard(const QString &label, QObject *parent = nullptr);

protected:
    void initialize() override;
    bool testConnection() override;

    // IOBoard interface
    bool configure(IOBoardConfig &config) override;
    std::map<int, double> readAnalogChannels() override;
    std::map<int, bool> readDigitalChannels() override;

    void sleep(bool b) override;
    void ioReadSettings() override;

private:
    QJsonObject configToJson(const IOBoardConfig &config) const;
    bool jsonToConfig(const QJsonObject &obj, IOBoardConfig &config) const;
};

#endif // PYTHONIOBOARD_H
