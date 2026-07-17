#ifndef PYTHONLASERFREQCONVERSIONSTAGE_H
#define PYTHONLASERFREQCONVERSIONSTAGE_H

#include <hardware/optional/laserfreqconversion/laserfreqconversionstage.h>
#include <hardware/core/hardwareregistry.h>

#include "pythonhardwarebase.h"

/*!
 * \brief LaserFreqConversionStage subclass that dispatches all virtual methods to a Python subprocess via IPC
 *
 * PythonLaserFreqConversionStage launches a Python subprocess (via
 * PythonProcess) that loads a user-written frequency-conversion-stage
 * driver script. The two private pure virtual methods required by
 * LaserFreqConversionStage — setPos() and readPos(), which move to and
 * read back a local input-beam wavenumber in cm⁻¹ — are translated to
 * JSON requests sent over stdin/stdout pipes.
 *
 * The LaserFreqConversionStage base class handles:
 *   - Move verification (readback tolerance / verify flag) in setPosition()
 *   - The registered conversionOp()/harmonicOrder() settings accessors
 *   - setPosition()/readPosition() dispatch
 *
 * PythonLaserFreqConversionStage only needs to implement initialize(),
 * testConnection(), settings reload, sleep, and the two IPC-dispatched
 * virtuals.
 */
class PythonLaserFreqConversionStage : public LaserFreqConversionStage, public PythonHardwareBase
{
    Q_OBJECT
public:
    explicit PythonLaserFreqConversionStage(const QString &label, QObject *parent = nullptr);

protected:
    void initialize() override;
    bool testConnection() override;
    void hwReadSettings() override;
    void sleep(bool b) override;

private:
    void setPos(double localCm1) override;
    double readPos() override;
};

#endif // PYTHONLASERFREQCONVERSIONSTAGE_H
