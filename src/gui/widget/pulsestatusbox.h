#ifndef PULSESTATUSBOX_H
#define PULSESTATUSBOX_H

#include "hardwarestatusbox.h"
#include <data/experiment/hardware/optional/pulsegenerator/pulsegenconfig.h>

class QLabel;
class Led;

class PulseStatusBox : public HardwareStatusBox
{
    Q_OBJECT
public:
    explicit PulseStatusBox(const QString &key, QWidget *parent = nullptr);

public slots:
    void updatePulseLeds(const QString k, const PulseGenConfig &cc);
    void updatePulseSetting(const QString k,int index,PulseGenConfig::Setting s, QVariant val);
    void rebuild();

signals:

private:
    std::vector<std::pair<QLabel*,Led*>> d_ledList;
    std::vector<QString> d_channelFullNames;
    int d_labelMaxWidth{0};
    QLabel *p_repLabel;
    Led *p_enLed;

    void updateAll();
    void updateChannelTooltip(int ch);
    void setChannelLabel(int ch, const QString &fullName);
    PulseGenConfig d_config;

};

#endif // PULSESTATUSBOX_H
