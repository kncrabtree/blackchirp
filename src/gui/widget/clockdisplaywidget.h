#ifndef CLOCKDISPLAYWIDGET_H
#define CLOCKDISPLAYWIDGET_H

#include <QWidget>

#include <data/experiment/rfconfig.h>

class QDoubleSpinBox;

class ClockDisplayWidget : public QWidget
{
    Q_OBJECT
public:
    explicit ClockDisplayWidget(QWidget *parent = nullptr);

signals:

public slots:
    void updateFrequency(RfConfig::ClockType t, double f);

private:
    QHash<RfConfig::ClockType,QDoubleSpinBox*> d_boxes;
};

#endif // CLOCKDISPLAYWIDGET_H
