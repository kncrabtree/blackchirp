#ifndef TEMPERATURECONTROLWIDGET_H
#define TEMPERATURECONTROLWIDGET_H

#include <QWidget>

#include <data/storage/settingsstorage.h>
#include <hardware/optional/tempcontroller/temperaturecontrollerconfig.h>

class QLineEdit;
class QPushButton;

namespace BC::Key::TCW {
static const QString key("TemperatureControlWidget");
static const QString channels("channels");
static const QString chName("name");
}

class TemperatureControlWidget : public QWidget, public SettingsStorage
{
    Q_OBJECT
public:
    struct TChannels {
        QLineEdit *le;
        QPushButton *button;
    };

    explicit TemperatureControlWidget(QWidget *parent = nullptr);
    ~TemperatureControlWidget();

signals:
    int channelNameChanged(int,QString);
    int channelEnableChanged(int,bool);

public slots:
    void setFromConfig(const TemperatureControllerConfig &cfg);
    void setChannelEnabled(int ch, bool en);

private:
    std::vector<TChannels> d_channelWidgets;

};

#endif // TEMPERATURECONTROLWIDGET_H
