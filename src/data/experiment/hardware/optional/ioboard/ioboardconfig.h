#ifndef IOBOARDCONFIG_H
#define IOBOARDCONFIG_H

#include <data/experiment/digitizerconfig.h>


namespace BC::Store::Digi {
inline constexpr QLatin1StringView chName{"Name"};
}

namespace BC::Aux::IOB {
inline const QString ain{"AnalogInput%1"};
inline const QString din{"DigitalInput%1"};
}

class IOBoardConfig : public DigitizerConfig
{
public:
    IOBoardConfig(const QString& hwKey);

    void setAnalogName(int ch, const QString name);
    void setDigitalName(int ch, const QString name);
    QString analogName(int ch) const;
    QString digitalName(int ch) const;

    // HeaderStorage interface
protected:
    void storeValues() override;
    void retrieveValues() override;

private:
    std::map<int,QString> d_analogNames;
    std::map<int,QString> d_digitalNames;
};

#endif // IOBOARDCONFIG_H
