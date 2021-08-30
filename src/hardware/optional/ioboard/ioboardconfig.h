#ifndef IOBOARDCONFIG_H
#define IOBOARDCONFIG_H

#include <data/experiment/digitizerconfig.h>


namespace BC::Store::Digi {
static const QString name{"Name"};
static const QString iob{"IOBoardDigitizer"};
}

namespace BC::Aux::IOB {
static const QString ain{"AnalogInput%1"};
static const QString din{"DigitalInput%1"};
}

class IOBoardConfig : public DigitizerConfig
{
public:
    IOBoardConfig();

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
