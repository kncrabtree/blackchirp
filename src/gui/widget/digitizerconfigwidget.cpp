#include "digitizerconfigwidget.h"

#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QFormLayout>
#include <QGroupBox>
#include <QComboBox>
#include <QSpinBox>
#include <QDoubleSpinBox>
#include <QCheckBox>
#include <QLabel>

#include <data/experiment/digitizerconfig.h>

using namespace BC::Key::Digi;
using namespace BC::Key::DigiWidget;

DigitizerConfigWidget::DigitizerConfigWidget(const QString widgetKey, const QString digHwKey, QWidget *parent) :
    QWidget(parent), SettingsStorage(widgetKey)
{

    auto hbl = new QHBoxLayout;

    SettingsStorage s(digHwKey,Hardware);

    auto chvbl = new QVBoxLayout;

    for(int i=0; i<s.get(numChannels,4); ++i)
    {
        auto chBox = new QGroupBox(QString("Ch ")+QString::number(i+1));
        chBox->setCheckable(true);

        auto fl = new QFormLayout;

        auto fsBox = new QDoubleSpinBox;
        fsBox->setDecimals(3);
        fsBox->setRange(s.get(minFullScale,1e-3),s.get(maxFullScale,5.0));
        fsBox->setPrefix("± ");
        fsBox->setSuffix(" V");
        fsBox->setSingleStep(fsBox->minimum());
        fsBox->setValue(getArrayValue(dwChannels,i,lFullScale,fsBox->minimum()));

        auto voBox = new QDoubleSpinBox;
        voBox->setDecimals(3);
        voBox->setRange(s.get(minVOffset,-5.0),s.get(maxVOffset,5.0));
        voBox->setSuffix(" V");
        voBox->setSingleStep((voBox->maximum() - voBox->minimum())/100.0);
        voBox->setValue(getArrayValue(dwChannels,i,lVOffset,0.0));
        chBox->setLayout(fl);

        connect(chBox,&QGroupBox::toggled,fsBox,&QDoubleSpinBox::setEnabled);
        connect(chBox,&QGroupBox::toggled,voBox,&QDoubleSpinBox::setEnabled);
        chBox->setChecked(getArrayValue(dwChannels,i,chEnabled,false));

        fl->addRow("Full Scale",fsBox);
        fl->addRow("Offset",voBox);

        for(int i=0; i<fl->rowCount(); ++i)
        {
            auto l = static_cast<QLabel*>(fl->itemAt(i,QFormLayout::LabelRole)->widget());
            l->setSizePolicy(QSizePolicy::Minimum,QSizePolicy::MinimumExpanding);
            l->setAlignment(Qt::AlignRight|Qt::AlignVCenter);
        }

        chBox->setLayout(fl);
        d_channelWidgets.append({chBox,fsBox,voBox});
        chvbl->addWidget(chBox,1);
    }

    chvbl->addSpacerItem(new QSpacerItem(1,1,QSizePolicy::Minimum,QSizePolicy::Expanding));
    hbl->addLayout(chvbl,1);

    auto vl = new QVBoxLayout;

    auto horBox = new QGroupBox("Data Transfer");
    auto hfl = new QFormLayout;

    p_recLengthBox = new QSpinBox;
    p_recLengthBox->setRange(1,s.get(maxRecordLength,__INT_MAX__));
    p_recLengthBox->setValue(get(lRecLen,1000));
    registerGetter(lRecLen,p_recLengthBox,&QSpinBox::value);

    p_sampleRateBox = new QComboBox;
    auto sr = s.getArray(sampleRates);
    if(sr.size() > 0)
    {
        for(auto m : sr)
        {
            auto txt = m.find(srText);
            auto val = m.find(srValue);
            if(txt != m.end() && val != m.end())
                p_sampleRateBox->addItem(txt->second.toString(),val->second);
        }
    }
    else
    {
        //this code is not tested!
        p_sampleRateBox->setEditable(true);
        auto v = new QDoubleValidator(this);
        v->setRange(1,1e11,0);
        v->setNotation(QDoubleValidator::ScientificNotation);
        p_sampleRateBox->setValidator(v);
    }
    p_sampleRateBox->setCurrentIndex(get(lSampIndex,-1));
    registerGetter(lSampIndex,p_sampleRateBox,&QComboBox::currentIndex);

    p_bytesPerPointBox = new QSpinBox;
    p_bytesPerPointBox->setRange(1,s.get(maxBytes,2));
    p_bytesPerPointBox->setValue(get(lBytes,1));
    registerGetter(lBytes,p_bytesPerPointBox,&QSpinBox::value);

    p_byteOrderBox = new QComboBox;
    p_byteOrderBox->addItem("Little Endian",QDataStream::LittleEndian);
    p_byteOrderBox->addItem("Big Endian",QDataStream::BigEndian);
    p_byteOrderBox->setCurrentIndex(get(lByteOrder,0));
    registerGetter(lByteOrder,p_byteOrderBox,&QComboBox::currentIndex);

    hfl->addRow("Record Length",p_recLengthBox);
    hfl->addRow("Sample Rate",p_sampleRateBox);
    hfl->addRow("Bytes Per Point",p_bytesPerPointBox);
    hfl->addRow("Byte Order",p_byteOrderBox);


    for(int i=0; i<hfl->rowCount(); ++i)
    {
        auto l = static_cast<QLabel*>(hfl->itemAt(i,QFormLayout::LabelRole)->widget());
        l->setSizePolicy(QSizePolicy::Minimum,QSizePolicy::MinimumExpanding);
        l->setAlignment(Qt::AlignRight|Qt::AlignVCenter);
    }
    horBox->setLayout(hfl);
    vl->addWidget(horBox,1);

    auto tBox = new QGroupBox("Trigger");
    auto tfl = new QFormLayout;

    p_triggerSourceBox = new QSpinBox;
    if(s.get(hasAuxChannel,true))
    {
        p_triggerSourceBox->setRange(0,s.get(numChannels,4));
        p_triggerSourceBox->setSpecialValueText("Aux");
    }
    else
        p_triggerSourceBox->setRange(1,s.get(numChannels,4));
    p_triggerSourceBox->setValue(get(lTrigSource,0));
    registerGetter(lTrigSource,p_triggerSourceBox,&QSpinBox::value);

    p_triggerSlopeBox = new QComboBox;
    p_triggerSlopeBox->addItem("Rising Edge",DigitizerConfig::RisingEdge);
    p_triggerSlopeBox->addItem("Falling Edge",DigitizerConfig::FallingEdge);
    p_triggerSlopeBox->setCurrentIndex(get(lTrigSlope,0));
    registerGetter(lTrigSlope,p_triggerSlopeBox,&QComboBox::currentIndex);

    p_triggerDelayBox = new QDoubleSpinBox;
    p_triggerDelayBox->setDecimals(6);
    p_triggerDelayBox->setSuffix(QString::fromUtf16(u" μs"));
    p_triggerDelayBox->setRange(s.get(minTrigDelay,-10.),s.get(maxTrigDelay,10.));
    p_triggerDelayBox->setValue(get(lTrigDelay,0.0));
    registerGetter(lTrigDelay,p_triggerDelayBox,&QDoubleSpinBox::value);

    p_triggerLevelBox = new QDoubleSpinBox;
    p_triggerLevelBox->setDecimals(3);
    p_triggerLevelBox->setSuffix(" V");
    p_triggerLevelBox->setRange(s.get(minTrigLevel,-5.),s.get(maxTrigLevel,5.));
    p_triggerLevelBox->setValue(get(lTrigLevel,0.0));
    p_triggerLevelBox->setSingleStep((p_triggerLevelBox->maximum()-p_triggerLevelBox->minimum())/100.0);
    registerGetter(lTrigLevel,p_triggerLevelBox,&QDoubleSpinBox::value);

    tfl->addRow("Source",p_triggerSourceBox);
    tfl->addRow("Slope",p_triggerSlopeBox);
    tfl->addRow("Delay",p_triggerDelayBox);
    tfl->addRow("Level",p_triggerLevelBox);
    for(int i=0; i<tfl->rowCount(); ++i)
    {
        auto l = static_cast<QLabel*>(tfl->itemAt(i,QFormLayout::LabelRole)->widget());
        l->setSizePolicy(QSizePolicy::Minimum,QSizePolicy::MinimumExpanding);
        l->setAlignment(Qt::AlignRight|Qt::AlignVCenter);
    }

    tBox->setLayout(tfl);
    vl->addWidget(tBox,1);

    auto aBox = new QGroupBox("Acquisition Setup");
    auto afl = new QFormLayout;

    p_blockAverageBox = new QCheckBox;
    p_blockAverageBox->setToolTip(QString(R"(If checked, the scope will acquire multiple records and return a single record containing the average.
On Tektronix scopes, this will be done using FastFrame with a summary frame.
For most scopes, this option is mutually exclusive with "Multiple Records" mode, which returns each individual record without averaging.)"));

    p_numAveragesBox = new QSpinBox;
    p_numAveragesBox->setRange(1,s.get(maxAverages,__INT_MAX__));
    p_numAveragesBox->setValue(get(lBlockNumAvg,1));
    p_numAveragesBox->setEnabled(false);
    p_numAveragesBox->setToolTip(QString(R"(Number of records to average. If 1, averaging will be disabled.
The actual number of records able to be averaged may be limited by the record length or data size.)"));
    registerGetter(lBlockNumAvg,p_numAveragesBox,&QSpinBox::value);

    p_multiRecordBox = new QCheckBox;
    p_multiRecordBox->setToolTip(QString(R"(If checked, the scope will acquire multiple records and return all of them at once.
On Tektronix scipes, this will be done using FastFrame mode with no summary frame.
For most scopes, this option is mutually exclusive with "Block Average" mode, which averages the individual records.)"));


    p_numRecordsBox = new QSpinBox;
    p_numRecordsBox->setRange(1,s.get(maxRecords,__INT_MAX__));
    p_numRecordsBox->setValue(get(lNumRecords,1));
    p_numRecordsBox->setEnabled(false);
    p_numRecordsBox->setToolTip(QString(R"(Number of records to acquire. If 1, this feature will be disabled.
The actual number of records able to be acquired may be limited by the record length or data size.)"));
    registerGetter(lNumRecords,p_numRecordsBox,&QSpinBox::value);


    if(!s.get(blockAverage,false))
    {
        p_blockAverageBox->setChecked(false);
        p_blockAverageBox->setEnabled(false);
        p_numAveragesBox->setValue(1);
        p_numAveragesBox->setEnabled(false);
    }

    if(!s.get(multiRecord,false))
    {
        p_multiRecordBox->setChecked(false);
        p_multiRecordBox->setEnabled(false);
        p_numRecordsBox->setValue(1);
        p_numRecordsBox->setEnabled(false);
    }

    if(!s.get(multiBlock,false))
    {
        connect(p_blockAverageBox,&QCheckBox::toggled,[this](bool en){
            p_numAveragesBox->setEnabled(en);
            p_multiRecordBox->setEnabled(!en);
            if(en)
                p_multiRecordBox->setChecked(false);
        });
        connect(p_multiRecordBox,&QCheckBox::toggled,[this](bool en){
            p_numRecordsBox->setEnabled(en);
            p_blockAverageBox->setEnabled(!en);
            if(en)
                p_blockAverageBox->setChecked(false);
        });
    }

    if(get(lBlockAvg,false))
        p_blockAverageBox->setChecked(true);
    if(get(lMultiRec,false))
        p_multiRecordBox->setChecked(true);

    registerGetter(lBlockAvg,static_cast<QAbstractButton*>(p_blockAverageBox),&QCheckBox::isChecked);
    registerGetter(lMultiRec,static_cast<QAbstractButton*>(p_multiRecordBox),&QCheckBox::isChecked);

    afl->addRow("Block Average",p_blockAverageBox);
    afl->addRow("# Averages",p_numAveragesBox);
    afl->addRow("Multiple Records",p_multiRecordBox);
    afl->addRow("# Records",p_numRecordsBox);

    for(int i=0; i<afl->rowCount(); ++i)
    {
        auto l = static_cast<QLabel*>(afl->itemAt(i,QFormLayout::LabelRole)->widget());
        l->setSizePolicy(QSizePolicy::Minimum,QSizePolicy::MinimumExpanding);
        l->setAlignment(Qt::AlignRight|Qt::AlignVCenter);
    }

    aBox->setLayout(afl);

    vl->addWidget(aBox);
    vl->addSpacerItem(new QSpacerItem(1,1,QSizePolicy::Minimum,QSizePolicy::Expanding));
    hbl->addLayout(vl,1);

    setLayout(hbl);
}

DigitizerConfigWidget::~DigitizerConfigWidget()
{
    std::vector<SettingsMap> l;
    l.reserve(d_channelWidgets.size());
    for(auto ch : d_channelWidgets)
    {
        l.push_back({
                        {chEnabled,ch.channelBox->isChecked()},
                        {lFullScale,ch.fullScaleBox->value()},
                        {lVOffset,ch.vOffsetBox->value()}
                    });
    }
    setArray(dwChannels,l,false);
}
