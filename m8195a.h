#ifndef M8195A_H
#define M8195A_H

#include "awg.h"

class M8195A : public AWG
{
    Q_OBJECT
public:
    explicit M8195A(QObject *parent=nullptr);

    // HardwareObject interface
public slots:
    void readSettings();
    Experiment prepareForExperiment(Experiment exp);
    void beginAcquisition();
    void endAcquisition();

protected:
    bool testConnection();
    void initialize();


private:
    bool m8195aWrite(const QString cmd);
};

#endif // M8195A_H
