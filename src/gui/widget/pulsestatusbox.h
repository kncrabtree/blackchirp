#ifndef PULSESTATUSBOX_H
#define PULSESTATUSBOX_H

#include <QGroupBox>

#include "hardwarestatusbox.h"
#include <hardware/optional/pulsegenerator/pulsegenconfig.h>

class QLabel;
class Led;

class PulseStatusBox : public HardwareStatusBox
{
    Q_OBJECT
public:
    explicit PulseStatusBox(QString key,QWidget *parent = nullptr);

public slots:
    void updatePulseLeds(const QString k, const PulseGenConfig &cc);
    void updatePulseSetting(const QString k,int index,PulseGenConfig::Setting s, QVariant val);

signals:

private:
    std::vector<std::pair<QLabel*,Led*>> d_ledList;
    QLabel *p_repLabel;
    Led *p_enLed;

    void updateAll();
    PulseGenConfig d_config;

};

#endif // PULSESTATUSBOX_H
