#ifndef M8195A_H
#define M8195A_H

#include <src/hardware/core/chirpsource/awg.h>

class M8195A : public AWG
{
    Q_OBJECT
public:
    explicit M8195A(QObject *parent=nullptr);

    // HardwareObject interface
public slots:
    void readSettings() override;
    Experiment prepareForExperiment(Experiment exp) override;
    void beginAcquisition() override;
    void endAcquisition() override;

protected:
    bool testConnection() override;
    void initialize() override;


private:
    bool m8195aWrite(const QString cmd);
};

#endif // M8195A_H
