#ifndef FTMWDIGITIZERCONFIG_H
#define FTMWDIGITIZERCONFIG_H

#include <data/experiment/digitizerconfig.h>

namespace  BC::Key::Digi {
static const QString ftmwDigi{"FtmwDigitizer"};
}

namespace BC::Store::Digi {
static const QString ftmwKey{"FtmwDigitizer"};
static const QString fidCh{"FidChannel"};
}

class FtmwDigitizerConfig : public DigitizerConfig
{
public:
    FtmwDigitizerConfig();

    int d_fidChannel{0};

    // HeaderStorage interface
protected:
    void storeValues() override;
    void retrieveValues() override;
};

#endif // FTMWDIGITIZERCONFIG_H
