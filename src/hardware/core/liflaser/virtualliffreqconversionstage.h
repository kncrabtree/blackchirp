#ifndef VIRTUALLIFFREQCONVERSIONSTAGE_H
#define VIRTUALLIFFREQCONVERSIONSTAGE_H

#include <hardware/core/liflaser/liffreqconversionstage.h>

class VirtualLifFreqConversionStage : public LifFreqConversionStage
{
    Q_OBJECT
public:
    VirtualLifFreqConversionStage(const QString& label, QObject *parent = nullptr);

protected:
    void initialize() override;
    bool testConnection() override;

    // LifFreqConversionStage interface
private:
    void setPos(double localCm1) override;
    double readPos() override;

    double d_pos;
};

#endif // VIRTUALLIFFREQCONVERSIONSTAGE_H
