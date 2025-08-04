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
    // TODO: Remove this legacy constructor after migrating all code to new label-based system
    // This constructor assumes compile-time hardware selection and old key structures
    FtmwDigitizerConfig(const QString subKey);
    
    // New label-based constructor
    FtmwDigitizerConfig(const QString& hwType, const QString& subKey, const QString& label);

    int d_fidChannel{0};

    // HeaderStorage interface
protected:
    void storeValues() override;
    void retrieveValues() override;
};

#endif // FTMWDIGITIZERCONFIG_H
