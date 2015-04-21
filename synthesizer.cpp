#include "synthesizer.h"

Synthesizer::Synthesizer(QObject *parent)
 : HardwareObject(parent)
{
    d_key = QString("synthesizer");
}

Synthesizer::~Synthesizer()
{

}
