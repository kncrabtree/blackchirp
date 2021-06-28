#ifndef FTMWDIGITIZERCONFIG_H
#define FTMWDIGITIZERCONFIG_H

#include <data/experiment/digitizerconfig.h>

namespace  BC::Key::Digi {
static const QString ftmwDigi("FtmwDigitizer");
}

class FtmwDigitizerConfig : public DigitizerConfig
{
public:
    FtmwDigitizerConfig();

    int d_fidChannel{0};
};

#endif // FTMWDIGITIZERCONFIG_H
