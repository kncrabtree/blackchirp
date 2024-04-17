#ifndef DIGITIZERCONFIGWIDGET_H
#define DIGITIZERCONFIGWIDGET_H

#include <QWidget>

#include <data/storage/settingsstorage.h>
#include <data/experiment/digitizerconfig.h>

class QGroupBox;
class QDoubleSpinBox;
class QSpinBox;
class QComboBox;
class QCheckBox;

namespace BC::Key::DigiWidget {
static const QString dwAnChannels{"channels"};
static const QString dwDigChannels{"digitalChannels"};
static const QString roleIndex{"roleIndex"};
static const QString digRead{"digitalRead"};
static const QString lFullScale{"lastFullScale"};
static const QString lVOffset{"lastVOffset"};
static const QString chEnabled{"enabled"};
static const QString lRecLen{"lastRecordLength"};
static const QString lSampIndex{"lastSampleRateIndex"};
static const QString lBytes{"lastBytesPerPoint"};
static const QString lByteOrder{"lastByteOrderIndex"};
static const QString lTrigSource{"lastTriggerSource"};
static const QString lTrigSlope{"lastTriggerSlopeIndex"};
static const QString lTrigDelay{"lastTriggerDelay"};
static const QString lTrigLevel{"lastTriggerLevel"};
static const QString lBlockAvg{"lastBlockAverageEnabled"};
static const QString lBlockNumAvg{"lastNumAverages"};
static const QString lMultiRec{"lastMultipleRecordsEnabled"};
static const QString lNumRecords{"lastNumMultipleRecords"};
}

class DigitizerConfigWidget : public QWidget, public SettingsStorage
{
    Q_OBJECT
public:
    explicit DigitizerConfigWidget(const QString widgetKey, const QString digHwKey, QWidget *parent = nullptr);
    virtual ~DigitizerConfigWidget();

    struct AnalogChannelWidgets {
        QGroupBox *channelBox;
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

signals:
    void edited();

public slots:
    void configureAnalogBoxes();

protected:
    QVector<AnalogChannelWidgets> d_anChannelWidgets;
    QVector<DigitalChannelWidgets> d_digChannelWidgets;

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
