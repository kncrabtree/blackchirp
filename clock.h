#ifndef CLOCK_H
#define CLOCK_H

#include "hardwareobject.h"

/**
 * @brief The Clock class defines an interface for an oscillator
 *
 * Unlike most other HardwareObjects, BlackChirp expects to have multiple clocks.
 * This creates
 *
 */

class Clock : public HardwareObject
{
    Q_OBJECT
public:
    explicit Clock(QObject *parent = nullptr);

    int numOutputs() { return d_numOutputs; }
    virtual QStringList channelNames() =0;

protected:
    int d_numOutputs;
};

#endif // CLOCK_H
