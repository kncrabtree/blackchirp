#ifndef DIGITIZERCONFIGWIDGET_H
#define DIGITIZERCONFIGWIDGET_H

#include <QWidget>

#include <data/storage/settingsstorage.h>

class QGroupBox;
class QDoubleSpinBox;
class QSpinBox;
class QComboBox;
class QCheckBox;

namespace BC::Key::DigiWidget {
static const QString dwChannels("channels");
static const QString lFullScale("lastFullScale");
static const QString lVOffset("lastVOffset");
static const QString chEnabled("enabled");
static const QString lRecLen("lastRecordLength");
static const QString lSampIndex("lastSampleRateIndex");
static const QString lBytes("lastBytesPerPoint");
static const QString lByteOrder("lastByteOrderIndex");
static const QString lTrigSource("lastTriggerSource");
static const QString lTrigSlope("lastTriggerSlopeIndex");
static const QString lTrigDelay("lastTriggerDelay");
static const QString lTrigLevel("lastTriggerLevel");
static const QString lBlockAvg("lastBlockAverageEnabled");
static const QString lBlockNumAvg("lastNumAverages");
static const QString lMultiRec("lastMultipleRecordsEnabled");
static const QString lNumRecords("lastNumMultipleRecords");
}

class DigitizerConfigWidget : public QWidget, public SettingsStorage
{
    Q_OBJECT
public:
    explicit DigitizerConfigWidget(const QString widgetKey, const QString digHwKey, QWidget *parent = nullptr);
    virtual ~DigitizerConfigWidget();

    struct ChannelWidgets {
        QGroupBox *channelBox;
        QDoubleSpinBox *fullScaleBox;
        QDoubleSpinBox *vOffsetBox;
    };

private:
    QList<ChannelWidgets> d_channelWidgets;

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


signals:

};

#endif // DIGITIZERCONFIGWIDGET_H
