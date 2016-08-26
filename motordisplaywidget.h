#ifndef MOTORDISPLAYWIDGET_H
#define MOTORDISPLAYWIDGET_H

#include <QWidget>

namespace Ui {
class MotorDisplayWidget;
}

class MotorDisplayWidget : public QWidget
{
    Q_OBJECT

public:
    explicit MotorDisplayWidget(QWidget *parent = 0);
    ~MotorDisplayWidget();

private:
    Ui::MotorDisplayWidget *ui;
};

#endif // MOTORDISPLAYWIDGET_H
