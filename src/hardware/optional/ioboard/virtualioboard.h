#ifndef VIRTUALIOBOARD_H
#define VIRTUALIOBOARD_H

#include <hardware/optional/ioboard/ioboard.h>

namespace BC::Key::IOB {
static const QString viobName("Virtual IO Board");
}

class VirtualIOBoard : public IOBoard
{
    Q_OBJECT
public:
    explicit VirtualIOBoard(QObject *parent = nullptr);

    // HardwareObject interface
protected:
    bool testConnection() override;
    void initialize() override;


    // IOBoard interface
protected:
    std::map<int, double> readAnalogChannels() override;
    std::map<int, bool> readDigitalChannels() override;
};

#endif // VIRTUALIOBOARD_H
