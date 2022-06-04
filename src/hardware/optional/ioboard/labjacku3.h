#ifndef LABJACKU3_H
#define LABJACKU3_H

#include <hardware/optional/ioboard/ioboard.h>

#include "u3.h"

namespace BC::Key::IOB {
static const QString labjacku3{"labjacku3"};
static const QString labjacku3Name("Labjack U3 IO Board");
static const QString serialNo{"serialNo"};
}

class LabjackU3 : public IOBoard
{
    Q_OBJECT
public:
    explicit LabjackU3(QObject *parent = nullptr);

private:
    HANDLE d_handle;
    u3CalibrationInfo d_calInfo;

    int d_serialNo;

    bool configure();
    void closeConnection();

    // HardwareObject interface
protected:
    bool testConnection() override;
    void initialize() override;

    // IOBoard interface
protected:
    std::map<int, double> readAnalogChannels() override;
    std::map<int, bool> readDigitalChannels() override;
};

#endif // LABJACKU3_H
