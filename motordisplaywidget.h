#ifndef MOTORDISPLAYWIDGET_H
#define MOTORDISPLAYWIDGET_H

#include <QWidget>

#include "experiment.h"
#include "motorscan.h"
#include "analysis.h"

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
    void updatePlots();

private:
    Ui::MotorDisplayWidget *ui;
    QList<MotorSliderWidget*> d_sliders;

    MotorScan d_currentScan;

    int d_winSize = 21, d_polyOrder = 3;
    bool d_smooth = true;
    Eigen::MatrixXd d_coefs;

};

#endif // MOTORDISPLAYWIDGET_H
