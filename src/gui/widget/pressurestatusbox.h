#ifndef PRESSURESTATUSBOX_H
#define PRESSURESTATUSBOX_H

#include "hardwarestatusbox.h"

#include <QString>

class QLabel;
class Led;

class PressureStatusBox : public HardwareStatusBox
{
    Q_OBJECT
public:
    PressureStatusBox(const QString &key, QWidget *parent = nullptr);

public slots:
    void pressureUpdate(const QString key,double p);
    void pressureControlUpdate(const QString key, bool en);
    void updateFromSettings();

private:
    QLabel *p_cpLabel;
    Led *p_led;
    int d_decimals{4};
    QString d_suffix;
};

#endif // PRESSURESTATUSBOX_H
