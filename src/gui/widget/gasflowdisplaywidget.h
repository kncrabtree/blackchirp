#ifndef GASFLOWDISPLAYWIDGET_H
#define GASFLOWDISPLAYWIDGET_H

#include "hardwarestatusbox.h"

#include <QVector>

class QLabel;
class QGridLayout;
class Led;

using FlowWidgets = std::tuple<QLabel*,QLabel*,Led*>;

class GasFlowDisplayBox : public HardwareStatusBox
{
    Q_OBJECT
public:
    explicit GasFlowDisplayBox(const QString key, QWidget *parent = nullptr);

public slots:
    void applySettings();
    void rebuild();
    void updateFlow(const QString key, int ch, double val);
    void updateFlowName(const QString key, int ch, const QString name);
    void updateFlowSetpoint(const QString key, int ch, double val);
    void updatePressureControl(const QString key, bool en);
    void updatePressure(const QString key, double p);

signals:

private:
    QVector<FlowWidgets> d_flowWidgets;
    QVector<int> d_channelDecimals;
    QVector<QString> d_channelSuffix;
    QVector<double> d_setpoints;

    QLabel *p_pressureLabel;
    Led *p_pressureLed;
    int d_pressureDecimals{3};
    QString d_pressureSuffix;

    void addChannelsToGrid(QGridLayout *gl);
    void updateSetpointTooltip(int ch);
};

#endif // GASFLOWDISPLAYWIDGET_H
