#ifndef MOTORSCANCONFIGWIDGET_H
#define MOTORSCANCONFIGWIDGET_H

#include <QWidget>

#include <modules/motor/data/motorscan.h>

namespace Ui {
class MotorScanConfigWidget;
}

class MotorScanConfigWidget : public QWidget
{
    Q_OBJECT

public:
    explicit MotorScanConfigWidget(QWidget *parent = 0);
    ~MotorScanConfigWidget();

    void setFromMotorScan(MotorScan ms);
    MotorScan toMotorScan();

public slots:
    void validateBoxes();
    bool validatePage();

private:
    Ui::MotorScanConfigWidget *ui;
};

#endif // MOTORSCANCONFIGWIDGET_H
