#ifndef LIFLASERWIDGET_H
#define LIFLASERWIDGET_H

#include <QWidget>

class QDoubleSpinBox;
class QPushButton;

class LifLaserWidget : public QWidget
{
    Q_OBJECT
public:
    explicit LifLaserWidget(QWidget *parent = nullptr);

signals:

private:
    QDoubleSpinBox *p_posBox;
    QPushButton *p_posSetButton;
    QPushButton *p_flButton;
};

#endif // LIFLASERWIDGET_H
