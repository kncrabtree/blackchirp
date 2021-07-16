#ifndef IOBOARDCONFIG_H
#define IOBOARDCONFIG_H

#include <data/experiment/digitizerconfig.h>

namespace BC::Key::Digi {
static const QString iob{"IOBoardDigitizer"};
}

namespace BC::Store::Digi {
static const QString anName{"AnalogNames"};
static const QString digName("DigitalNames");
static const QString name{"Name"};
}

namespace BC::Aux::IOB {
static const QString ain{"ain%1"};
static const QString din{"din%1"};
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
    void prepareToSave() override;
    void loadComplete() override;

private:
    std::map<int,QString> d_analogNames;
    std::map<int,QString> d_digitalNames;
};

#endif // IOBOARDCONFIG_H
