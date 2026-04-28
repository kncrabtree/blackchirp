#ifndef LABJACKU3_H
#define LABJACKU3_H

#include <hardware/optional/ioboard/ioboard.h>
#include <hardware/library/labjackdriver.h>

namespace BC::Key::IOB {
inline constexpr QLatin1StringView labjacku3{"labjacku3"};
inline const QString labjacku3Name{"Labjack U3 IO Board"};
inline constexpr QLatin1StringView serialNo{"serialNo"};
}

class LabjackU3 : public IOBoard
{
    Q_OBJECT
public:
    explicit LabjackU3(const QString& label, QObject *parent = nullptr);

private:
    BC::Labjack::HandlePtr d_handle;
    int d_serialNo;

    bool configureTimers();
    void closeConnection();

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

#endif // LABJACKU3_H
