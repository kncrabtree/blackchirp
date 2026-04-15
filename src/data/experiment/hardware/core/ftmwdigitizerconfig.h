#ifndef FTMWDIGITIZERCONFIG_H
#define FTMWDIGITIZERCONFIG_H

#include <data/experiment/digitizerconfig.h>

namespace BC::Store::Digi {
inline constexpr QLatin1StringView fidCh{"FidChannel"};
}

class FtmwDigitizerConfig : public DigitizerConfig
{
public:
    FtmwDigitizerConfig(const QString& hwKey);

    int d_fidChannel{0};

    // HeaderStorage interface
protected:
    void storeValues() override;
    void retrieveValues() override;
};

#endif // FTMWDIGITIZERCONFIG_H
