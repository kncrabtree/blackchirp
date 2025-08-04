#ifndef FTMWDIGITIZERCONFIG_H
#define FTMWDIGITIZERCONFIG_H

#include <data/experiment/digitizerconfig.h>
#include <data/settings/hardwarekeys.h>

namespace BC::Store::Digi {
static const QString fidCh{"FidChannel"};
}

class FtmwDigitizerConfig : public DigitizerConfig
{
public:
    FtmwDigitizerConfig(const QString& hwType, const QString& impl, const QString& label);

    int d_fidChannel{0};

    // HeaderStorage interface
protected:
    void storeValues() override;
    void retrieveValues() override;
};

#endif // FTMWDIGITIZERCONFIG_H
