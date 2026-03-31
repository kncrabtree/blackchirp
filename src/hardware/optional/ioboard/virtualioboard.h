#ifndef VIRTUALIOBOARD_H
#define VIRTUALIOBOARD_H

#include <hardware/optional/ioboard/ioboard.h>


class VirtualIOBoard : public IOBoard
{
    Q_OBJECT
public:
    explicit VirtualIOBoard(const QString& label, QObject *parent = nullptr);

    // HardwareObject interface
protected:
    bool testConnection() override;
    void initialize() override;


    // IOBoard interface
protected:
    bool configure(IOBoardConfig &config) override;
    std::map<int, double> readAnalogChannels() override;
    std::map<int, bool> readDigitalChannels() override;
};

#endif // VIRTUALIOBOARD_H
