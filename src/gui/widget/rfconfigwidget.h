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
inline constexpr QLatin1StringView key{"RfConfigWidget"};
inline constexpr QLatin1StringView awgM{"awgMultFactor"};
inline constexpr QLatin1StringView chirpM{"chirpMultFactor"};
inline constexpr QLatin1StringView upSB{"upconversionSideband"};
inline constexpr QLatin1StringView downSB{"downconversionSideband"};
inline constexpr QLatin1StringView comLO{"commonUpDownLO"};
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
    void applyClocks(QHash<RfConfig::ClockType, RfConfig::ClockFreq>);

private:
    Ui::RfConfigWidget *ui;
    ClockTableModel *p_ctm;

    // QWidget interface
public:
    QSize sizeHint() const override;
};

#endif // RFCONFIGWIDGET_H
