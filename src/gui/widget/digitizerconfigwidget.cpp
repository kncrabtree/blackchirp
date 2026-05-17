#include "digitizerconfigwidget.h"

#include <climits>

#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QGroupBox>
#include <QComboBox>
#include <QSpinBox>
#include <QDoubleSpinBox>
#include <QCheckBox>
#include <QLineEdit>
#include <QTableWidget>
#include <QHeaderView>

#include <data/experiment/digitizerconfig.h>
#include <data/settings/hardwarekeys.h>
#include <gui/widget/cellwidgethelpers.h>
#include <gui/widget/settingstable.h>

using namespace BC::Key::Digi;
using namespace BC::Store::Digi;
using namespace Qt::StringLiterals;
using BC::Gui::centerCellWidget;

DigitizerConfigWidget::DigitizerConfigWidget(const QString widgetKey, const QString digHwKey, bool withChannelNames, QWidget *parent) :
    QWidget(parent), SettingsStorage(widgetKey+"."+digHwKey), d_namesEnabled(withChannelNames), d_hwKey(digHwKey)
{
    using namespace BC::Key::DigiWidget;

    SettingsStorage s(d_hwKey,Hardware);

    auto centerCombo = [](QComboBox *cb) {
        if(!cb->isEditable())
        {
            cb->setEditable(true);
            cb->lineEdit()->setReadOnly(true);
        }
        cb->lineEdit()->setAlignment(Qt::AlignCenter);
        for(int i=0; i<cb->count(); ++i)
            cb->setItemData(i,Qt::AlignCenter,Qt::TextAlignmentRole);
    };

    auto numAn = s.get(numAnalogChannels,4);
    const int anCols = d_namesEnabled ? 4 : 3;
    const int anNameCol = 3;
    p_anTable = new QTableWidget(numAn,anCols,this);
    QStringList anHeaders{"Enable","Full Scale","Offset"};
    if(d_namesEnabled) anHeaders << "Name";
    p_anTable->setHorizontalHeaderLabels(anHeaders);
    p_anTable->horizontalHeader()->setSectionResizeMode(0,QHeaderView::ResizeToContents);
    if(d_namesEnabled)
    {
        p_anTable->horizontalHeader()->setSectionResizeMode(1,QHeaderView::ResizeToContents);
        p_anTable->horizontalHeader()->setSectionResizeMode(2,QHeaderView::ResizeToContents);
        p_anTable->horizontalHeader()->setSectionResizeMode(anNameCol,QHeaderView::Stretch);
    }
    else
    {
        p_anTable->horizontalHeader()->setSectionResizeMode(1,QHeaderView::Stretch);
        p_anTable->horizontalHeader()->setSectionResizeMode(2,QHeaderView::Stretch);
    }
    p_anTable->verticalHeader()->setSectionResizeMode(QHeaderView::ResizeToContents);
    p_anTable->setSelectionMode(QAbstractItemView::NoSelection);
    p_anTable->setFocusPolicy(Qt::NoFocus);
    QStringList anRowLabels;

    for(int i=0; i<numAn; ++i)
    {
        anRowLabels << QString("Ch %1").arg(i+1);

        auto chBox = new QCheckBox;
        chBox->setChecked(s.getArrayValue(dwAnChannels,i,en,false));

        auto fsBox = new QDoubleSpinBox;
        fsBox->setDecimals(3);
        fsBox->setRange(s.get(minFullScale,1e-3),s.get(maxFullScale,5.0));
        fsBox->setPrefix("± ");
        fsBox->setSuffix(" V");
        fsBox->setSingleStep(fsBox->minimum());
        fsBox->setValue(s.getArrayValue(dwAnChannels,i,fs,fsBox->minimum()));
        fsBox->setAlignment(Qt::AlignCenter);
        if(!d_namesEnabled)
            fsBox->setSizePolicy(QSizePolicy::Expanding,QSizePolicy::Fixed);

        auto voBox = new QDoubleSpinBox;
        voBox->setDecimals(3);
        voBox->setRange(s.get(minVOffset,-5.0),s.get(maxVOffset,5.0));
        voBox->setSuffix(" V");
        voBox->setSingleStep((voBox->maximum() - voBox->minimum())/100.0);
        voBox->setValue(s.getArrayValue(dwAnChannels,i,offset,0.0));
        voBox->setAlignment(Qt::AlignCenter);
        if(!d_namesEnabled)
            voBox->setSizePolicy(QSizePolicy::Expanding,QSizePolicy::Fixed);

        connect(chBox,&QCheckBox::toggled,fsBox,&QDoubleSpinBox::setEnabled);
        connect(chBox,&QCheckBox::toggled,voBox,&QDoubleSpinBox::setEnabled);
        connect(chBox,&QCheckBox::toggled,this,&DigitizerConfigWidget::configureAnalogBoxes);
        connect(chBox,&QCheckBox::toggled,this,&DigitizerConfigWidget::edited);
        connect(fsBox,&QDoubleSpinBox::valueChanged,this,&DigitizerConfigWidget::edited);
        connect(voBox,&QDoubleSpinBox::valueChanged,this,&DigitizerConfigWidget::edited);

        fsBox->setEnabled(chBox->isChecked());
        voBox->setEnabled(chBox->isChecked());

        centerCellWidget(p_anTable,i,0,chBox);
        p_anTable->setCellWidget(i,1,fsBox);
        p_anTable->setCellWidget(i,2,voBox);

        if(d_namesEnabled)
        {
            auto nameItem = new QTableWidgetItem(getArrayValue(dwAnChannels,i,channelName,QString("")));
            p_anTable->setItem(i,anNameCol,nameItem);
        }

        d_anChannelWidgets.insert({i+1,{chBox,fsBox,voBox}});
    }
    p_anTable->setVerticalHeaderLabels(anRowLabels);

    auto anBox = new QGroupBox("Analog Channels",this);
    auto anLayout = new QVBoxLayout;
    anLayout->addWidget(p_anTable);
    anBox->setLayout(anLayout);

    auto channelsHbl = new QHBoxLayout;
    channelsHbl->addWidget(anBox,1);

    int dch = s.get(numDigitalChannels,0);
    if(dch > 0)
    {
        const int digCols = d_namesEnabled ? 3 : 2;
        const int digNameCol = 2;
        p_digTable = new QTableWidget(dch,digCols,this);
        QStringList digHeaders{"Read","Role"};
        if(d_namesEnabled) digHeaders << "Name";
        p_digTable->setHorizontalHeaderLabels(digHeaders);
        p_digTable->horizontalHeader()->setSectionResizeMode(0,QHeaderView::ResizeToContents);
        if(d_namesEnabled)
        {
            p_digTable->horizontalHeader()->setSectionResizeMode(1,QHeaderView::ResizeToContents);
            p_digTable->horizontalHeader()->setSectionResizeMode(digNameCol,QHeaderView::Stretch);
        }
        else
            p_digTable->horizontalHeader()->setSectionResizeMode(1,QHeaderView::Stretch);
        p_digTable->verticalHeader()->setSectionResizeMode(QHeaderView::ResizeToContents);
        p_digTable->setSelectionMode(QAbstractItemView::NoSelection);
        p_digTable->setFocusPolicy(Qt::NoFocus);
        QStringList digRowLabels;

        for(int i=0; i<dch; ++i)
        {
            digRowLabels << QString("Ch %1").arg(i+1);

            auto readBox = new QCheckBox;
            readBox->setChecked(s.getArrayValue(dwDigChannels,i,digInp,false));

            auto roleBox = new QComboBox;
            roleBox->setEnabled(!readBox->isChecked());
            centerCombo(roleBox);

            connect(readBox,&QCheckBox::toggled,roleBox,&QComboBox::setDisabled);
            connect(readBox,&QCheckBox::toggled,this,&DigitizerConfigWidget::edited);

            centerCellWidget(p_digTable,i,0,readBox);
            p_digTable->setCellWidget(i,1,roleBox);

            if(d_namesEnabled)
            {
                auto nameItem = new QTableWidgetItem(getArrayValue(dwDigChannels,i,channelName,QString("")));
                p_digTable->setItem(i,digNameCol,nameItem);
            }

            d_digChannelWidgets.insert({i+1,{readBox,roleBox}});
        }
        p_digTable->setVerticalHeaderLabels(digRowLabels);

        auto digBox = new QGroupBox("Digital Channels",this);
        auto digLayout = new QVBoxLayout;
        digLayout->addWidget(p_digTable);
        digBox->setLayout(digLayout);
        channelsHbl->addWidget(digBox,1);
    }

    auto bottomHbl = new QHBoxLayout;

    auto horBox = new QGroupBox("Data Transfer");

    p_recLengthBox = new QSpinBox;
    p_recLengthBox->setRange(1,s.get(maxRecordLength,INT_MAX));
    p_recLengthBox->setValue(s.get(recLen,1));
    p_recLengthBox->setAlignment(Qt::AlignCenter);

    p_sampleRateBox = new QComboBox;
    auto sr = s.getArray(sampleRates);
    double samp = s.get(sRate,0.0);
    int idx = -1;
    for(auto m : sr)
    {
        auto txt = m.find(srText);
        auto val = m.find(srValue);
        if(txt != m.end() && val != m.end())
        {
            p_sampleRateBox->addItem(txt->second.toString(),val->second);
            if(qFuzzyCompare(val->second.toDouble(),samp))
                idx = p_sampleRateBox->count()-1;
        }
    }
    p_sampleRateBox->setCurrentIndex(idx);

    p_bytesPerPointBox = new QSpinBox;
    p_bytesPerPointBox->setRange(1,s.get(maxBytes,2));
    p_bytesPerPointBox->setValue(s.get(bpp,1));
    p_bytesPerPointBox->setAlignment(Qt::AlignCenter);

    p_byteOrderBox = new QComboBox;
    p_byteOrderBox->addItem("Little Endian",DigitizerConfig::LittleEndian);
    p_byteOrderBox->addItem("Big Endian",DigitizerConfig::BigEndian);
    p_byteOrderBox->setCurrentIndex(p_byteOrderBox->findData(s.get(bo)));

    centerCombo(p_sampleRateBox);
    centerCombo(p_byteOrderBox);

    auto dtTable = new SettingsTable(this);
    dtTable->setFocusPolicy(Qt::NoFocus);
    dtTable->addSettingRow("Record Length"_L1,p_recLengthBox);
    dtTable->addSettingRow("Sample Rate"_L1,p_sampleRateBox);
    dtTable->addSettingRow("Bytes Per Point"_L1,p_bytesPerPointBox);
    dtTable->addSettingRow("Byte Order"_L1,p_byteOrderBox);

    auto horLayout = new QVBoxLayout;
    horLayout->addWidget(dtTable);
    horBox->setLayout(horLayout);
    bottomHbl->addWidget(horBox,1);

    auto tBox = new QGroupBox("Trigger");

    p_triggerSourceBox = new QSpinBox;
    if(s.get(hasAuxTriggerChannel,true))
    {
        p_triggerSourceBox->setRange(0,s.get(numAnalogChannels,4));
        p_triggerSourceBox->setSpecialValueText("Aux");
    }
    else
        p_triggerSourceBox->setRange(1,s.get(numAnalogChannels,4));
    p_triggerSourceBox->setValue(s.get(trigCh,0));
    p_triggerSourceBox->setAlignment(Qt::AlignCenter);

    p_triggerSlopeBox = new QComboBox;
    p_triggerSlopeBox->addItem("Rising Edge",DigitizerConfig::RisingEdge);
    p_triggerSlopeBox->addItem("Falling Edge",DigitizerConfig::FallingEdge);
    p_triggerSlopeBox->setCurrentIndex(p_triggerSlopeBox->findData(s.get(trigSlope)));
    centerCombo(p_triggerSlopeBox);

    p_triggerDelayBox = new QDoubleSpinBox;
    p_triggerDelayBox->setDecimals(6);
    p_triggerDelayBox->setSuffix(QString::fromUtf16(u" μs"));
    p_triggerDelayBox->setRange(s.get(minTrigDelay,-10.),s.get(maxTrigDelay,10.));
    p_triggerDelayBox->setValue(s.get(trigDelay,0.0));
    p_triggerDelayBox->setAlignment(Qt::AlignCenter);

    p_triggerLevelBox = new QDoubleSpinBox;
    p_triggerLevelBox->setDecimals(3);
    p_triggerLevelBox->setSuffix(" V");
    p_triggerLevelBox->setRange(s.get(minTrigLevel,-5.),s.get(maxTrigLevel,5.));
    p_triggerLevelBox->setValue(s.get(trigLevel,0.0));
    p_triggerLevelBox->setSingleStep((p_triggerLevelBox->maximum()-p_triggerLevelBox->minimum())/100.0);
    p_triggerLevelBox->setAlignment(Qt::AlignCenter);

    auto trigTable = new SettingsTable(this);
    trigTable->setFocusPolicy(Qt::NoFocus);
    trigTable->addSettingRow("Source"_L1,p_triggerSourceBox);
    trigTable->addSettingRow("Slope"_L1,p_triggerSlopeBox);
    trigTable->addSettingRow("Delay"_L1,p_triggerDelayBox);
    trigTable->addSettingRow("Level"_L1,p_triggerLevelBox);

    auto tLayout = new QVBoxLayout;
    tLayout->addWidget(trigTable);
    tBox->setLayout(tLayout);
    bottomHbl->addWidget(tBox,1);

    if(!s.get(isTriggered,true))
        tBox->setEnabled(false);

    auto aBox = new QGroupBox("Acquisition Setup");

    p_blockAverageBox = new QCheckBox;
    p_blockAverageBox->setToolTip(QString(R"(If checked, the scope will acquire multiple records and return a single record containing the average.
On Tektronix scopes, this will be done using FastFrame with a summary frame.
For most scopes, this option is mutually exclusive with "Multiple Records" mode, which returns each individual record without averaging.)"));

    p_numAveragesBox = new QSpinBox;
    p_numAveragesBox->setRange(1,s.get(maxAverages,INT_MAX));
    p_numAveragesBox->setValue(s.get(numAvg,1));
    p_numAveragesBox->setEnabled(false);
    p_numAveragesBox->setAlignment(Qt::AlignCenter);
    p_numAveragesBox->setToolTip(QString(R"(Number of records to average. If 1, averaging will be disabled.
The actual number of records able to be averaged may be limited by the record length or data size.)"));

    p_multiRecordBox = new QCheckBox;
    p_multiRecordBox->setToolTip(QString(R"(If checked, the scope will acquire multiple records and return all of them at once.
On Tektronix scipes, this will be done using FastFrame mode with no summary frame.
For most scopes, this option is mutually exclusive with "Block Average" mode, which averages the individual records.)"));


    p_numRecordsBox = new QSpinBox;
    p_numRecordsBox->setRange(1,s.get(maxRecords,INT_MAX));
    p_numRecordsBox->setValue(s.get(multiRecNum,1));
    p_numRecordsBox->setEnabled(false);
    p_numRecordsBox->setAlignment(Qt::AlignCenter);
    p_numRecordsBox->setToolTip(QString(R"(Number of records to acquire. If 1, this feature will be disabled.
The actual number of records able to be acquired may be limited by the record length or data size.)"));

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

    if(s.get(blockAvg,false))
        p_blockAverageBox->setChecked(true);
    if(s.get(multiRec,false))
        p_multiRecordBox->setChecked(true);

    auto aTable = new QTableWidget(2,2,this);
    aTable->setHorizontalHeaderLabels({"On","Count"});
    aTable->setVerticalHeaderLabels({"Block Average","Multiple Records"});
    aTable->horizontalHeader()->setSectionResizeMode(0,QHeaderView::ResizeToContents);
    aTable->horizontalHeader()->setSectionResizeMode(1,QHeaderView::Stretch);
    aTable->verticalHeader()->setSectionResizeMode(QHeaderView::ResizeToContents);
    aTable->setSelectionMode(QAbstractItemView::NoSelection);
    aTable->setFocusPolicy(Qt::NoFocus);
    centerCellWidget(aTable,0,0,p_blockAverageBox);
    aTable->setCellWidget(0,1,p_numAveragesBox);
    centerCellWidget(aTable,1,0,p_multiRecordBox);
    aTable->setCellWidget(1,1,p_numRecordsBox);

    auto aLayout = new QVBoxLayout;
    aLayout->addWidget(aTable);
    aBox->setLayout(aLayout);

    bottomHbl->addWidget(aBox,1);
    if(!s.get(canBlockAverage,false) && !s.get(canMultiRecord,false))
        aBox->hide();

    auto outerVbl = new QVBoxLayout;
    outerVbl->addLayout(channelsHbl,1);
    outerVbl->addLayout(bottomHbl,0);

    connect(p_recLengthBox,&QSpinBox::valueChanged,this,&DigitizerConfigWidget::edited);
    connect(p_sampleRateBox,&QComboBox::currentIndexChanged,this,&DigitizerConfigWidget::edited);
    connect(p_bytesPerPointBox,&QSpinBox::valueChanged,this,&DigitizerConfigWidget::edited);
    connect(p_byteOrderBox,&QComboBox::currentIndexChanged,this,&DigitizerConfigWidget::edited);
    connect(p_triggerSourceBox,&QSpinBox::valueChanged,this,&DigitizerConfigWidget::edited);
    connect(p_triggerSlopeBox,&QComboBox::currentIndexChanged,this,&DigitizerConfigWidget::edited);
    connect(p_triggerDelayBox,&QDoubleSpinBox::valueChanged,this,&DigitizerConfigWidget::edited);
    connect(p_triggerLevelBox,&QDoubleSpinBox::valueChanged,this,&DigitizerConfigWidget::edited);
    connect(p_blockAverageBox,&QCheckBox::toggled,this,&DigitizerConfigWidget::edited);
    connect(p_numAveragesBox,&QSpinBox::valueChanged,this,&DigitizerConfigWidget::edited);
    connect(p_multiRecordBox,&QCheckBox::toggled,this,&DigitizerConfigWidget::edited);
    connect(p_numRecordsBox,&QSpinBox::valueChanged,this,&DigitizerConfigWidget::edited);

    setLayout(outerVbl);
    configureAnalogBoxes();
}

DigitizerConfigWidget::~DigitizerConfigWidget()
{
    if(!d_namesEnabled)
        return;

    using namespace BC::Key::DigiWidget;

    if(p_anTable)
    {
        const int col = p_anTable->columnCount() - 1;
        for(int i=0; i<p_anTable->rowCount(); ++i)
        {
            auto text = p_anTable->item(i,col)->text();
            if((std::size_t) i == getArraySize(dwAnChannels))
                appendArrayMap(dwAnChannels,{{channelName,text}});
            else
                setArrayValue(dwAnChannels,i,channelName,text);
        }
    }

    if(p_digTable)
    {
        const int col = p_digTable->columnCount() - 1;
        for(int i=0; i<p_digTable->rowCount(); ++i)
        {
            auto text = p_digTable->item(i,col)->text();
            if((std::size_t) i == getArraySize(dwDigChannels))
                appendArrayMap(dwDigChannels,{{channelName,text}});
            else
                setArrayValue(dwDigChannels,i,channelName,text);
        }
    }
}

QString DigitizerConfigWidget::analogChannelName(int channel) const
{
    if(!d_namesEnabled || !p_anTable)
        return {};
    const int row = channel - 1;
    if(row < 0 || row >= p_anTable->rowCount())
        return {};
    auto item = p_anTable->item(row,p_anTable->columnCount()-1);
    return item ? item->text() : QString();
}

void DigitizerConfigWidget::setAnalogChannelName(int channel, const QString &name)
{
    if(!d_namesEnabled || !p_anTable)
        return;
    const int row = channel - 1;
    if(row < 0 || row >= p_anTable->rowCount())
        return;
    auto item = p_anTable->item(row,p_anTable->columnCount()-1);
    if(item)
        item->setText(name);
}

QString DigitizerConfigWidget::digitalChannelName(int channel) const
{
    if(!d_namesEnabled || !p_digTable)
        return {};
    const int row = channel - 1;
    if(row < 0 || row >= p_digTable->rowCount())
        return {};
    auto item = p_digTable->item(row,p_digTable->columnCount()-1);
    return item ? item->text() : QString();
}

void DigitizerConfigWidget::setDigitalChannelName(int channel, const QString &name)
{
    if(!d_namesEnabled || !p_digTable)
        return;
    const int row = channel - 1;
    if(row < 0 || row >= p_digTable->rowCount())
        return;
    auto item = p_digTable->item(row,p_digTable->columnCount()-1);
    if(item)
        item->setText(name);
}

int DigitizerConfigWidget::numAnalogChecked() const
{
    int out = 0;
    for(auto &[_,ch] : d_anChannelWidgets)
    {
        if(ch.channelBox->isChecked())
            ++out;
    }

    return out;
}

int DigitizerConfigWidget::numDigitalChecked() const
{
    return 0;
}

bool DigitizerConfigWidget::multiRecordChecked() const
{
    return p_multiRecordBox->isChecked();
}

bool DigitizerConfigWidget::blockAverageChecked() const
{
    return p_blockAverageBox->isChecked();
}

int DigitizerConfigWidget::numAverages() const
{
    return p_numAveragesBox->value();
}

int DigitizerConfigWidget::numRecords() const
{
    return p_numRecordsBox->value();
}

void DigitizerConfigWidget::setFromConfig(const DigitizerConfig &c)
{
    for(auto &[k,ch] : c.d_analogChannels)
    {
        auto it = d_anChannelWidgets.find(k);
        if(it != d_anChannelWidgets.end())
        {
            it->second.channelBox->setChecked(ch.enabled);
            it->second.vOffsetBox->setValue(ch.offset);
            it->second.fullScaleBox->setValue(ch.fullScale);
        }
    }


    for(auto &[k,ch] : c.d_digitalChannels)
    {
        auto it = d_digChannelWidgets.find(k);
        if(it != d_digChannelWidgets.end())
        {
            it->second.readBox->setChecked(ch.input);
        }
    }

     p_triggerSourceBox->setValue(c.d_triggerChannel);
     p_triggerDelayBox->setValue(c.d_triggerDelayUSec);
     p_triggerLevelBox->setValue(c.d_triggerLevel);
     p_triggerSlopeBox->setCurrentIndex(p_triggerSlopeBox->findData(c.d_triggerSlope));

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
    for(auto &[k,ch] : d_anChannelWidgets)
    {
        c.d_analogChannels.insert_or_assign(
                    k,DigitizerConfig::AnalogChannel{ch.channelBox->isChecked(),
                                                     ch.fullScaleBox->value(),
                                                     ch.vOffsetBox->value()});
    }

    c.d_digitalChannels.clear();
    for(auto &[k,ch] : d_digChannelWidgets)
    {
        if(ch.readBox->isChecked())
            c.d_digitalChannels.insert_or_assign(k,DigitizerConfig::DigitalChannel{true,true,-1});
        else
        {
            if(ch.roleBox->currentIndex() >= 0 && ch.roleBox->currentData().toInt() >= 0)
                c.d_digitalChannels.insert_or_assign(k,DigitizerConfig::DigitalChannel{false,false,ch.roleBox->currentData().toInt()});
            else
                c.d_digitalChannels.insert_or_assign(k,DigitizerConfig::DigitalChannel{false,false,-1});
        }
    }

    c.d_triggerChannel = p_triggerSourceBox->value();
    c.d_triggerLevel = p_triggerLevelBox->value();
    c.d_triggerSlope = p_triggerSlopeBox->currentData().value<DigitizerConfig::TriggerSlope>();
    c.d_triggerDelayUSec = p_triggerDelayBox->value();

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
        for(auto &[_,ch] : d_anChannelWidgets)
            ch.channelBox->setEnabled(true);
    }
    else if(checked == d_maxAnalogEnabled)
    {
        for(auto &[_,ch] : d_anChannelWidgets)
            ch.channelBox->setEnabled(ch.channelBox->isChecked());
    }
    else
    {
        checked = 0;
        for(auto &[_,ch] : d_anChannelWidgets)
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
