#ifndef RFCONFIGWIDGET_H
#define RFCONFIGWIDGET_H

#include <QWidget>
#include <data/storage/settingsstorage.h>

#include <data/experiment/rfconfig.h>
#include <data/model/clocktablemodel.h>


namespace Ui {
class RfConfigWidget;
}

namespace BC::Key::RfConfigWidget {
static const QString key{"RfConfigWidget"};
static const QString awgM{"awgMultFactor"};
static const QString chirpM{"chirpMultFactor"};
static const QString upSB{"upconversionSideband"};
static const QString downSB{"downconversionSideband"};
static const QString comLO{"commonUpDownLO"};
}

class RfConfigWidget : public QWidget, public SettingsStorage
{
    Q_OBJECT

public:
    explicit RfConfigWidget(QWidget *parent = 0);
    ~RfConfigWidget();

    void setFromRfConfig(const RfConfig &c);
    void setClocks(const QHash<RfConfig::ClockType, RfConfig::ClockFreq> c);
    void toRfConfig(RfConfig &c);
    QString getHwKey(RfConfig::ClockType type) const;
    bool commonLO() const;

signals:
    void edited();

private:
    Ui::RfConfigWidget *ui;
    ClockTableModel *p_ctm;
};

#endif // RFCONFIGWIDGET_H
