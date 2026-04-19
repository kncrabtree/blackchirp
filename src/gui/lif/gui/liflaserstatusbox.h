#ifndef LIFLASERSTATUSBOX_H
#define LIFLASERSTATUSBOX_H

#include <QString>

#include <gui/widget/hardwarestatusbox.h>

class QLabel;
class Led;

class LifLaserStatusBox : public HardwareStatusBox
{
    Q_OBJECT
public:
    LifLaserStatusBox(const QString &key, QWidget *parent = nullptr);

    void applySettings();
    void setPosition(double d);
    void setFlashlampEnabled(bool en);

private:
    QLabel *p_posLabel;
    Led *p_led;
    int d_decimals{2};
    QString d_suffix;
    double d_position{0.0};

};

#endif // LIFLASERSTATUSBOX_H
