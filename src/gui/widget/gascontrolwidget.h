#ifndef GASCONTROLWIDGET_H
#define GASCONTROLWIDGET_H

#include <QWidget>
#include <QVector>

#include <data/storage/settingsstorage.h>
#include <data/experiment/hardware/optional/flowcontroller/flowconfig.h>

class QCheckBox;
class QDoubleSpinBox;
class QTableWidget;

namespace BC::Key::GasControl {
inline constexpr QLatin1StringView key{"GasControlWidget"};
}

class GasControlWidget : public QWidget, public SettingsStorage
{
    Q_OBJECT
public:
    struct ChannelWidgets {
        QDoubleSpinBox *setpointBox;
        QCheckBox *enableBox;
    };

    explicit GasControlWidget(const FlowConfig &cfg, QWidget *parent = nullptr);
    ~GasControlWidget() {}
    FlowConfig &toConfig();

public slots:
    void applySettings();
    void updateGasSetpoint(const QString key, int i, double sp);
    void updateChannelEnabled(const QString key, int i, bool en);
    void updatePressureSetpoint(const QString key, double sp);
    void updatePressureControl(const QString key, bool en);

signals:
    void nameUpdate(QString,int,QString);
    void gasSetpointUpdate(QString,int,double);
    void enableUpdate(QString,int,bool);
    void pressureSetpointUpdate(QString,double);
    void pressureControlUpdate(QString,bool);

private:
    void initialize(const FlowConfig &cfg);

    FlowConfig d_config;
    QTableWidget *p_table{nullptr};
    QVector<ChannelWidgets> d_widgets;
    QDoubleSpinBox *p_pressureSetpointBox;
    QCheckBox *p_pressureControlBox;
};

#endif // GASCONTROLWIDGET_H
