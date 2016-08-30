#ifndef MOTORSLIDERWIDGET_H
#define MOTORSLIDERWIDGET_H

#include <QWidget>

#include <QLabel>
#include <QSlider>
#include <QDoubleSpinBox>

#include "motorscan.h"

class MotorSliderWidget : public QWidget
{
    Q_OBJECT
public:
    explicit MotorSliderWidget(QWidget *parent = 0);

signals:
    void valueChanged(int);

public slots:
    void setRange(double min, double max, int steps, int decimals = 2);
    void updateSlider(double newVal);
    void updateBox(int newVal);
    void setAxis(MotorScan::MotorDataAxis a);

protected:
    QLabel *p_label;
    QSlider *p_slider;
    QDoubleSpinBox *p_dsb;

    double d_min, d_max, d_stepSize;
    int d_numSteps;

    void setLabel(QString s);
    void setUnits(QString u);
};

#endif // MOTORSLIDERWIDGET_H
