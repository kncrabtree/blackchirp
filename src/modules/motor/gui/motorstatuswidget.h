#ifndef MOTORSTATUSWIDGET_H
#define MOTORSTATUSWIDGET_H

#include <QWidget>

class Led;
class QLabel;
class QProgressBar;

#include <src/data/experiment/experiment.h>

namespace Ui {
class MotorStatusWidget;
}

class MotorStatusWidget : public QWidget
{
    Q_OBJECT
public:
    explicit MotorStatusWidget(QWidget *parent = 0);
    ~MotorStatusWidget();
    struct AxisWidget {
        QLabel *label;
        Led *negLimLed;
        QProgressBar *positionBar;
        Led *posLimLed;
        double minPos;
        double maxPos;
        double currentPos;
    };

    void prepareForExperiment(const Experiment e);

public slots:
    void updateRanges();
    void updatePosition(MotorScan::MotorAxis axis, double pos);
    void updateLimit(MotorScan::MotorAxis axis, bool n, bool p);
    void updateProgress(int s);

private:
    Ui::MotorStatusWidget *ui;

    AxisWidget d_x, d_y, d_z;
};

#endif // MOTORSTATUSWIDGET_H
