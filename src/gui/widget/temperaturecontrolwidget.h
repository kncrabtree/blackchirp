#ifndef TEMPERATURECONTROLWIDGET_H
#define TEMPERATURECONTROLWIDGET_H

#include <QWidget>

#include <data/storage/settingsstorage.h>
#include <data/experiment/hardware/optional/tempcontroller/temperaturecontrollerconfig.h>

class QCheckBox;
class QTableWidget;

namespace BC::Key::TCW {
inline constexpr QLatin1StringView key{"TemperatureControlWidget"};
}

class TemperatureControlWidget : public QWidget, public SettingsStorage
{
    Q_OBJECT
public:
    struct TChannels {
        QCheckBox *checkBox;
    };

    explicit TemperatureControlWidget(const TemperatureControllerConfig &cfg, QWidget *parent = nullptr);
    ~TemperatureControlWidget();
    TemperatureControllerConfig &toConfig();

signals:
    int channelNameChanged(QString,uint,QString);
    int channelEnableChanged(QString,uint,bool);

public slots:
    void setFromConfig(const TemperatureControllerConfig &cfg);
    void setChannelEnabled(const QString key, uint ch, bool en);

private:
    QTableWidget *p_table{nullptr};
    std::vector<TChannels> d_channelWidgets;
    TemperatureControllerConfig d_config;

};

#endif // TEMPERATURECONTROLWIDGET_H
