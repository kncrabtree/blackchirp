#ifndef PRESSURECONTROLWIDGET_H
#define PRESSURECONTROLWIDGET_H

#include <QWidget>

#include <hardware/optional/pressurecontroller/pressurecontrollerconfig.h>

class QDoubleSpinBox;
class QPushButton;

class PressureControlWidget : public QWidget
{
    Q_OBJECT
public:
    explicit PressureControlWidget(const PressureControllerConfig &cfg, QWidget *parent = nullptr);

    PressureControllerConfig &toConfig();

signals:
    void setpointChanged(QString,double);
    void pressureControlModeChanged(QString,bool);
    void valveOpen(QString);
    void valveClose(QString);

public slots:
    void pressureSetpointUpdate(const QString key, double p);
    void pressureControlModeUpdate(const QString key, bool en);

private:
    QDoubleSpinBox *p_setpointBox;
    QPushButton *p_controlButton;

    PressureControllerConfig d_config;

};

#endif // PRESSURECONTROLWIDGET_H
