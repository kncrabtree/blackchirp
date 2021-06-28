#include "digitizerconfig.h"

#include <cmath>

DigitizerConfig::DigitizerConfig()
{

}

double DigitizerConfig::xIncr() const
{
    if(d_sampleRate > 0.0)
        return 1.0/d_sampleRate;

    return nan("");
}

double DigitizerConfig::yMult(int ch) const
{
    if(ch < 0 || ch >= d_channels.size())
        return nan("");

    return d_channels.at(ch).fullScale * static_cast<double>( 2 << (d_bytesPerPoint*8 - 1));
}
