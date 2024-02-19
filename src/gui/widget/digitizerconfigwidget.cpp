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
#include <QScrollArea>

#include <data/experiment/digitizerconfig.h>

using namespace BC::Key::Digi;
using namespace BC::Key::DigiWidget;

DigitizerConfigWidget::DigitizerConfigWidget(const QString widgetKey, const QString digHwKey, QWidget *parent) :
    QWidget(parent), SettingsStorage(widgetKey), d_hwKey(digHwKey)
{

    auto hbl = new QHBoxLayout;

    SettingsStorage s(d_hwKey,Hardware);

    auto sa = new QScrollArea;
    auto anContainer = new QWidget;
    auto chvbl = new QVBoxLayout;

    for(int i=0; i<s.get(numAnalogChannels,4); ++i)
    {
        auto chBox = new QGroupBox(QString("Analog Ch ")+QString::number(i+1));
        chBox->setCheckable(true);

        auto fl = new QFormLayout;

        auto fsBox = new QDoubleSpinBox;
        fsBox->setDecimals(3);
        fsBox->setRange(s.get(minFullScale,1e-3),s.get(maxFullScale,5.0));
        fsBox->setPrefix("± ");
        fsBox->setSuffix(" V");
        fsBox->setSingleStep(fsBox->minimum());
        fsBox->setValue(getArrayValue(dwAnChannels,i,lFullScale,fsBox->minimum()));

        auto voBox = new QDoubleSpinBox;
        voBox->setDecimals(3);
        voBox->setRange(s.get(minVOffset,-5.0),s.get(maxVOffset,5.0));
        voBox->setSuffix(" V");
        voBox->setSingleStep((voBox->maximum() - voBox->minimum())/100.0);
        voBox->setValue(getArrayValue(dwAnChannels,i,lVOffset,0.0));
        chBox->setLayout(fl);

        connect(chBox,&QGroupBox::toggled,fsBox,&QDoubleSpinBox::setEnabled);
        connect(chBox,&QGroupBox::toggled,voBox,&QDoubleSpinBox::setEnabled);
        connect(chBox,&QGroupBox::toggled,this,&DigitizerConfigWidget::configureAnalogBoxes);
        connect(chBox,&QGroupBox::toggled,this,&DigitizerConfigWidget::edited);
        chBox->setChecked(getArrayValue(dwAnChannels,i,chEnabled,false));

        fl->addRow("Full Scale",fsBox);
        fl->addRow("Offset",voBox);

        for(int i=0; i<fl->rowCount(); ++i)
        {
            auto l = static_cast<QLabel*>(fl->itemAt(i,QFormLayout::LabelRole)->widget());
            l->setSizePolicy(QSizePolicy::Minimum,QSizePolicy::MinimumExpanding);
            l->setAlignment(Qt::AlignRight|Qt::AlignVCenter);
        }

        chBox->setLayout(fl);
        d_anChannelWidgets.append({chBox,fsBox,voBox});
        chvbl->addWidget(chBox,1);
    }

    chvbl->addSpacerItem(new QSpacerItem(1,1,QSizePolicy::Minimum,QSizePolicy::Expanding));
    anContainer->setLayout(chvbl);
    sa->setWidget(anContainer);
    sa->setWidgetResizable(true);
    hbl->addWidget(sa,1);

    int dch = s.get(numDigitalChannels,0);
    if(dch > 0)
    {
        auto dchvbl = new QVBoxLayout;
        auto digBox = new QGroupBox("Digital Channels");
        auto innervlb = new QVBoxLayout;
        auto dgl = new QGridLayout;
        dgl->addWidget(new QLabel("Channel"),0,0,1,1,Qt::AlignCenter);
        dgl->addWidget(new QLabel("Read?"),0,1,1,1,Qt::AlignCenter);
        dgl->addWidget(new QLabel("Role"),0,2,1,1,Qt::AlignCenter);

        for(int i=0; i<dch; ++i)
        {
            auto lbl = new QLabel(QString::number(i+1));
            lbl->setSizePolicy(QSizePolicy::Minimum,QSizePolicy::MinimumExpanding);
            lbl->setAlignment(Qt::AlignRight|Qt::AlignVCenter);
            dgl->addWidget(lbl,i+1,0,1,1,Qt::AlignCenter);

            auto readBox = new QCheckBox;
            readBox->setChecked(getArrayValue(dwDigChannels,i,chEnabled,false));
            dgl->addWidget(readBox,i+1,1,1,1,Qt::AlignCenter);

            auto roleBox = new QComboBox;
            if(readBox->isChecked())
                roleBox->setEnabled(false);
            dgl->addWidget(roleBox,i+1,2,1,1,Qt::AlignCenter);

            connect(readBox,&QCheckBox::toggled,roleBox,&QComboBox::setDisabled);

            d_digChannelWidgets.append({readBox,roleBox});
        }

        dgl->setColumnStretch(0,0);
        dgl->setColumnStretch(1,0);
        dgl->setColumnStretch(2,1);

        innervlb->addLayout(dgl,0);
        innervlb->addSpacerItem(new QSpacerItem(1,1,QSizePolicy::Minimum,QSizePolicy::Expanding));

        digBox->setLayout(innervlb);
        dchvbl->addWidget(digBox);
        hbl->addWidget(digBox);
    }


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
    p_byteOrderBox->addItem("Little Endian",DigitizerConfig::LittleEndian);
    p_byteOrderBox->addItem("Big Endian",DigitizerConfig::BigEndian);
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
    if(s.get(hasAuxTriggerChannel,true))
    {
        p_triggerSourceBox->setRange(0,s.get(numAnalogChannels,4));
        p_triggerSourceBox->setSpecialValueText("Aux");
    }
    else
        p_triggerSourceBox->setRange(1,s.get(numAnalogChannels,4));
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

    if(!s.get(isTriggered,true))
        tBox->setEnabled(false);

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


    if(!s.get(canBlockAverage,false))
    {
        p_blockAverageBox->setChecked(false);
        p_blockAverageBox->setEnabled(false);
        p_numAveragesBox->setValue(1);
        p_numAveragesBox->setEnabled(false);
    }

    if(!s.get(canMultiRecord,false))
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
    else
    {
        p_multiRecordBox->setEnabled(true);
        p_numRecordsBox->setEnabled(true);
        p_blockAverageBox->setEnabled(true);
        p_numAveragesBox->setEnabled(true);
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
    if(!s.get(canBlockAverage,false) && !s.get(canMultiRecord,false))
        aBox->hide();
    vl->addSpacerItem(new QSpacerItem(1,1,QSizePolicy::Minimum,QSizePolicy::Expanding));
    hbl->addLayout(vl,1);

    setLayout(hbl);
    configureAnalogBoxes();
}

DigitizerConfigWidget::~DigitizerConfigWidget()
{
    for(int i=0; i < d_anChannelWidgets.size(); ++i)
    {
        auto &ch = d_anChannelWidgets.at(i);
        SettingsMap m{
            {chEnabled,ch.channelBox->isChecked()},
            {lFullScale,ch.fullScaleBox->value()},
            {lVOffset,ch.vOffsetBox->value()}
        };

        if((std::size_t)i == getArraySize(dwAnChannels))
            appendArrayMap(dwAnChannels,m);
        else
        {
            for(auto &[k,v] : m)
                setArrayValue(dwAnChannels,i,k,v);
        }
    }

    for(int i=0; i<d_digChannelWidgets.size(); ++i)
    {
        auto &ch = d_digChannelWidgets.at(i);
        SettingsMap m{
            {digRead,ch.readBox->isChecked()},
            {roleIndex,ch.roleBox->currentIndex()}
        };

        if((std::size_t) i == getArraySize(dwDigChannels))
            appendArrayMap(dwDigChannels,m);
        else
        {
            for(auto &[k,v] : m)
                setArrayValue(dwDigChannels,i,k,v);
        }
    }
}

int DigitizerConfigWidget::numAnalogChecked()
{
    int out = 0;
    for(auto &ch : d_anChannelWidgets)
    {
        if(ch.channelBox->isChecked())
            ++out;
    }

    return out;
}

int DigitizerConfigWidget::numDigitalChecked()
{
    return 0;
}

void DigitizerConfigWidget::setFromConfig(const DigitizerConfig &c)
{
    for(auto &ch : c.d_analogChannels)
    {
        d_anChannelWidgets[ch.first-1].channelBox->setChecked(true);
        d_anChannelWidgets[ch.first-1].vOffsetBox->setValue(ch.second.offset);
        d_anChannelWidgets[ch.first-1].fullScaleBox->setValue(ch.second.fullScale);
    }


    for(auto &ch : c.d_digitalChannels)
        d_digChannelWidgets[ch.first-1].readBox->setChecked(ch.second.input);

     p_triggerSourceBox->setValue(c.d_triggerChannel);
     p_triggerDelayBox->setValue(c.d_triggerDelayUSec);
     p_triggerLevelBox->setValue(c.d_triggerLevel);
     p_triggerSlopeBox->setCurrentIndex(p_triggerSlopeBox->findData(c.d_triggerSlope));

     if(p_sampleRateBox->isEditable())
         p_sampleRateBox->setEditText(QString::number(c.d_sampleRate,'e',6));
     else
         p_sampleRateBox->setCurrentIndex(p_sampleRateBox->findData(QVariant(c.d_sampleRate)));
     p_recLengthBox->setValue(c.d_recordLength);

     p_bytesPerPointBox->setValue(c.d_bytesPerPoint);
     p_byteOrderBox->setCurrentIndex(p_byteOrderBox->findData(c.d_byteOrder));

     SettingsStorage s(d_hwKey,Hardware);
     if(s.get(canBlockAverage,false))
     {
         p_blockAverageBox->setChecked(c.d_blockAverage);
         p_numAveragesBox->setValue(c.d_numAverages);
     }
     else
         p_numAveragesBox->setValue(1);

     if(s.get(canMultiRecord,false))
     {
         p_multiRecordBox->setChecked(c.d_multiRecord);
         p_numRecordsBox->setValue(c.d_numRecords);
     }
     else
         p_numRecordsBox->setValue(1);
}

void DigitizerConfigWidget::toConfig(DigitizerConfig &c)
{
    c.d_analogChannels.clear();
    for(int i=0; i<d_anChannelWidgets.size(); ++i)
    {
        auto &ch = d_anChannelWidgets.at(i);
        if(ch.channelBox->isChecked())
            c.d_analogChannels.insert_or_assign(i+1,DigitizerConfig::AnalogChannel{ch.fullScaleBox->value(),
                                                                                 ch.vOffsetBox->value()});
    }

    c.d_digitalChannels.clear();
    for(int i=0; i<d_digChannelWidgets.size(); ++i)
    {
        auto &ch = d_digChannelWidgets.at(i);
        if(ch.readBox->isChecked())
            c.d_digitalChannels.insert_or_assign(i+1,DigitizerConfig::DigitalChannel{true,-1});
        else
        {
            if(ch.roleBox->currentIndex() >= 0 && ch.roleBox->currentData().toInt() >= 0)
                c.d_digitalChannels.insert_or_assign(i+1,DigitizerConfig::DigitalChannel{false,ch.roleBox->currentData().toInt()});
        }
    }

    c.d_triggerChannel = p_triggerSourceBox->value();
    c.d_triggerLevel = p_triggerLevelBox->value();
    c.d_triggerSlope = p_triggerSlopeBox->currentData().value<DigitizerConfig::TriggerSlope>();
    c.d_triggerDelayUSec = p_triggerDelayBox->value();

    if(p_sampleRateBox->isEditable())
        c.d_sampleRate = p_sampleRateBox->currentText().toDouble();
    else
        c.d_sampleRate = p_sampleRateBox->currentData().toDouble();
    c.d_recordLength = p_recLengthBox->value();

    c.d_bytesPerPoint = p_bytesPerPointBox->value();
    c.d_byteOrder = p_byteOrderBox->currentData().value<DigitizerConfig::ByteOrder>();

    c.d_blockAverage = p_blockAverageBox->isChecked();
    if(c.d_blockAverage)
        c.d_numAverages = p_numAveragesBox->value();
    else
    {
        p_numAveragesBox->setValue(1);
        c.d_numAverages = 1;
    }

    c.d_multiRecord = p_multiRecordBox->isChecked();
    if(c.d_multiRecord)
        c.d_numRecords = p_numRecordsBox->value();
    else
    {
        p_numRecordsBox->setValue(1);
        c.d_numRecords = 1;
    }
}

void DigitizerConfigWidget::configureAnalogBoxes()
{
    if(d_maxAnalogEnabled < 0)
        return;

    int checked = numAnalogChecked();
    if(checked < d_maxAnalogEnabled)
    {
        for(auto &ch : d_anChannelWidgets)
            ch.channelBox->setEnabled(true);
    }
    else if(checked == d_maxAnalogEnabled)
    {
        for(auto &ch : d_anChannelWidgets)
            ch.channelBox->setEnabled(ch.channelBox->isChecked());
    }
    else
    {
        checked = 0;
        for(auto &ch : d_anChannelWidgets)
        {
            if(checked == d_maxAnalogEnabled)
            {
                ch.channelBox->setChecked(false);
                ch.channelBox->setEnabled(false);
            }
            else
            {
                ch.channelBox->setEnabled(ch.channelBox->isChecked());
                if(ch.channelBox->isChecked())
                    ++checked;
            }
        }
    }
}
