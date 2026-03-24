#ifndef TEMPERATURESTATUSBOX_H
#define TEMPERATURESTATUSBOX_H

#include "hardwarestatusbox.h"

class QLabel;
class QDoubleSpinBox;

class TemperatureStatusBox : public HardwareStatusBox
{
    Q_OBJECT
public:
    struct ChWidgets {
        QLabel *label;
        QDoubleSpinBox *box;
        bool active{false};
    };

    TemperatureStatusBox(const QString key, QWidget *parent = nullptr);

public slots:
    void loadFromSettings();
    void setTemperature(const QString key, uint ch, double t);
    void setChannelName(const QString key, uint ch, const QString name);
    void setChannelEnabled(const QString key, uint ch, bool en);

private:
    void updateNoActiveLabel();

    std::vector<ChWidgets> d_widgets;
    QLabel *p_noActiveLabel;
};

#endif // TEMPERATURESTATUSBOX_H
