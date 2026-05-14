#ifndef DIGITIZERCONFIGWIDGET_H
#define DIGITIZERCONFIGWIDGET_H

#include <QWidget>

#include <data/storage/settingsstorage.h>
#include <data/experiment/digitizerconfig.h>

class QDoubleSpinBox;
class QSpinBox;
class QComboBox;
class QCheckBox;
class QTableWidget;

namespace BC::Key::DigiWidget {
inline constexpr QLatin1StringView channelName{"name"};
}

class DigitizerConfigWidget : public QWidget, public SettingsStorage
{
    Q_OBJECT
public:
    explicit DigitizerConfigWidget(const QString widgetKey, const QString digHwKey, bool withChannelNames = false, QWidget *parent = nullptr);
    virtual ~DigitizerConfigWidget();

    struct AnalogChannelWidgets {
        QCheckBox *channelBox;
        QDoubleSpinBox *fullScaleBox;
        QDoubleSpinBox *vOffsetBox;
    };

    struct DigitalChannelWidgets {
        QCheckBox *readBox;
        QComboBox *roleBox;
    };

    int d_maxAnalogEnabled{-1};
    int d_maxDigitalEnabled{-1};

    int numAnalogChecked() const;
    int numDigitalChecked() const;
    bool multiRecordChecked() const;
    bool blockAverageChecked() const;
    int numAverages() const;
    int numRecords() const;

    void setFromConfig(const DigitizerConfig &c);
    void toConfig(DigitizerConfig &c);

    bool channelNamesEnabled() const { return d_namesEnabled; }
    QString analogChannelName(int channel) const;
    void setAnalogChannelName(int channel, const QString &name);
    QString digitalChannelName(int channel) const;
    void setDigitalChannelName(int channel, const QString &name);

signals:
    void edited();

public slots:
    void configureAnalogBoxes();

protected:
    std::map<int,AnalogChannelWidgets> d_anChannelWidgets;
    std::map<int,DigitalChannelWidgets> d_digChannelWidgets;

    QTableWidget *p_anTable{nullptr};
    QTableWidget *p_digTable{nullptr};
    bool d_namesEnabled{false};

    QSpinBox *p_triggerSourceBox;
    QComboBox *p_triggerSlopeBox;
    QDoubleSpinBox *p_triggerDelayBox;
    QDoubleSpinBox *p_triggerLevelBox;

    QSpinBox *p_recLengthBox;
    QComboBox *p_sampleRateBox;
    QSpinBox *p_bytesPerPointBox;
    QComboBox *p_byteOrderBox;

    QCheckBox *p_blockAverageBox;
    QSpinBox *p_numAveragesBox;
    QCheckBox *p_multiRecordBox;
    QSpinBox *p_numRecordsBox;

    const QString d_hwKey;

};

#endif // DIGITIZERCONFIGWIDGET_H
