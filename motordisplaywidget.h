#ifndef MOTORDISPLAYWIDGET_H
#define MOTORDISPLAYWIDGET_H

#include <QWidget>

#include "experiment.h"
#include "motorscan.h"

class MotorSliderWidget;

namespace Ui {
class MotorDisplayWidget;
}

class MotorDisplayWidget : public QWidget
{
    Q_OBJECT

public:
    explicit MotorDisplayWidget(QWidget *parent = 0);
    ~MotorDisplayWidget();

public slots:
    void prepareForScan(const MotorScan s);
    void newMotorData(const MotorScan s);

private:
    Ui::MotorDisplayWidget *ui;
    QList<MotorSliderWidget*> d_sliders;

    MotorScan d_currentScan;
};

#endif // MOTORDISPLAYWIDGET_H
