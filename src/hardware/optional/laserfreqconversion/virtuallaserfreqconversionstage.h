#ifndef VIRTUALLASERFREQCONVERSIONSTAGE_H
#define VIRTUALLASERFREQCONVERSIONSTAGE_H

#include <hardware/optional/laserfreqconversion/laserfreqconversionstage.h>

class VirtualLaserFreqConversionStage : public LaserFreqConversionStage
{
    Q_OBJECT
public:
    VirtualLaserFreqConversionStage(const QString& label, QObject *parent = nullptr);

    //! Accepts a trim so trim sweeps can be exercised without hardware; the trim has no simulated effect.
    bool supportsTrim() const override { return true; }

protected:
    void initialize() override;
    bool testConnection() override;

    // LaserFreqConversionStage interface
private:
    void setPos(double localCm1) override;
    double readPos() override;

    double d_pos;
};

#endif // VIRTUALLASERFREQCONVERSIONSTAGE_H
