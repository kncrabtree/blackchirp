#include "pulseconfigwidget.h"

#include <QCheckBox>
#include <QComboBox>
#include <QDoubleSpinBox>
#include <QGridLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QHeaderView>
#include <QLabel>
#include <QMessageBox>
#include <QMetaEnum>
#include <QPushButton>
#include <QSpinBox>
#include <QStandardItem>
#include <QStandardItemModel>
#include <QTableWidget>
#include <QTabWidget>
#include <QToolButton>
#include <QVBoxLayout>

#include <gui/widget/cellwidgethelpers.h>
#include <gui/plot/pulseplot.h>
#include <gui/style/themecolors.h>
#include <hardware/optional/pulsegenerator/pulsegenerator.h>
#include <hardware/optional/chirpsource/awg.h>

using BC::Gui::centerCellWidget;

namespace {
constexpr int StdEnabledCol  = 0;
constexpr int StdDelayCol    = 1;
constexpr int StdWidthCol    = 2;
constexpr int StdInvCol      = 3;
constexpr int StdRoleCol     = 4;
constexpr int StdNameCol     = 5;
constexpr int StdNumCols     = 6;

constexpr int AdvSyncCol      = 0;
constexpr int AdvModeCol      = 1;
constexpr int AdvDutyOnCol    = 2;
constexpr int AdvDutyOffCol   = 3;
constexpr int AdvDelayStepCol = 4;
constexpr int AdvWidthStepCol = 5;
constexpr int AdvNumCols      = 6;

const QString lockedWhilePulsingTip{
    "Locked while pulsing is enabled. Toggle Pulsing Enabled off "
    "in System Settings to change this setting."};

QStringList channelRowLabels(int n)
{
    QStringList out;
    out.reserve(n);
    for(int i=0; i<n; ++i)
        out << QString::number(i+1);
    return out;
}

void applyOnIcon(QToolButton *b, bool on)
{
    auto role = on ? ThemeColors::StatusSuccess : ThemeColors::IconSecondary;
    b->setIcon(ThemeColors::createThemedIcon(QString(":/icons/power.svg"), role, b));
}
}

PulseConfigWidget::PulseConfigWidget(const PulseGenConfig &cfg, QWidget *parent) :
    QWidget(parent),
    SettingsStorage(BC::Key::widgetKey(BC::Key::PulseWidget::key,cfg.headerKey())),
    d_key{cfg.headerKey()}
{
    ps_config = std::make_shared<PulseGenConfig>(cfg);
    auto vc = qOverload<double>(&QDoubleSpinBox::valueChanged);

    int numChannels = cfg.d_channels.size();
    if(!containsArray(BC::Key::PulseWidget::channels))
        setArray(BC::Key::PulseWidget::channels,{});

    SettingsStorage hwSettings(cfg.headerKey(), SettingsStorage::Hardware);
    const bool canDisable = hwSettings.get(BC::Key::PGen::canDisableChannels, true);

    auto hbl = new QHBoxLayout(this);

    auto leftSide = new QVBoxLayout;
    hbl->addLayout(leftSide, 1);

    // ---- System Settings groupbox ----
    p_mainGb = new QGroupBox("System Settings", this);
    auto mainGb = p_mainGb;
    auto gl = new QGridLayout(mainGb);

    auto pulsingLabel = new QLabel("Pulsing Enabled:", mainGb);
    pulsingLabel->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
    gl->addWidget(pulsingLabel, 0, 0);

    p_sysOnOffButton = new QPushButton("Off", mainGb);
    p_sysOnOffButton->setCheckable(true);
    p_sysOnOffButton->setChecked(false);
    p_sysOnOffButton->setMinimumHeight(32);
    connect(p_sysOnOffButton, &QPushButton::toggled, this, [this](bool en){
        emit changeSetting(d_key, -1, PulseGenConfig::PGenEnabledSetting, en);
        p_sysOnOffButton->setText(en ? QString("On") : QString("Off"));
        for(auto &ch : d_widgetList)
        {
            ch.modeBox->setDisabled(en || ch.locked);
            ch.syncBox->setDisabled(en || ch.locked);
        }
        p_sysModeBox->setDisabled(en);
    });
    gl->addWidget(p_sysOnOffButton, 0, 1, 1, 3);

    auto modeLabel = new QLabel("Pulse Mode:", mainGb);
    modeLabel->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
    gl->addWidget(modeLabel, 1, 0);

    p_sysModeBox = new EnumComboBox<PulseGenConfig::PGenMode>(mainGb);
    connect(p_sysModeBox, qOverload<int>(&QComboBox::currentIndexChanged), this, [this](int i){
        emit changeSetting(d_key, -1, PulseGenConfig::PGenModeSetting, p_sysModeBox->value(i));
        p_repRateBox->setEnabled(p_sysModeBox->value(i) == PulseGenConfig::Continuous);
    });
    gl->addWidget(p_sysModeBox, 1, 1);

    auto rateLabel = new QLabel("Rep Rate:", mainGb);
    rateLabel->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
    gl->addWidget(rateLabel, 1, 2);

    p_repRateBox = new QDoubleSpinBox(mainGb);
    p_repRateBox->setSuffix(QString(" Hz"));
    connect(p_repRateBox, vc, this, [this](double val){
        emit changeSetting(d_key, -1, PulseGenConfig::RepRateSetting, val);
    });
    gl->addWidget(p_repRateBox, 1, 3);

    gl->setColumnStretch(0, 0);
    gl->setColumnStretch(1, 1);
    gl->setColumnStretch(2, 0);
    gl->setColumnStretch(3, 1);

    leftSide->addWidget(mainGb);

    // ---- Channel tables ----
    auto tabs = new QTabWidget(this);

    p_standardTable = new QTableWidget(numChannels, StdNumCols, tabs);
    p_standardTable->setHorizontalHeaderLabels(
        {"On", "Delay", "Width", "Inv?", "Role", "Name"});
    for(int c=0; c<StdNumCols; ++c)
        p_standardTable->horizontalHeader()->setSectionResizeMode(c,
            (c == StdNameCol) ? QHeaderView::Stretch : QHeaderView::ResizeToContents);
    p_standardTable->verticalHeader()->setSectionResizeMode(QHeaderView::ResizeToContents);
    p_standardTable->setVerticalHeaderLabels(channelRowLabels(numChannels));
    p_standardTable->setSelectionMode(QAbstractItemView::NoSelection);
    p_standardTable->setFocusPolicy(Qt::NoFocus);

    p_advancedTable = new QTableWidget(numChannels, AdvNumCols, tabs);
    p_advancedTable->setHorizontalHeaderLabels(
        {"Sync", "Mode", "Duty On", "Duty Off", "Delay Step", "Width Step"});
    for(int c=0; c<AdvNumCols; ++c)
        p_advancedTable->horizontalHeader()->setSectionResizeMode(c, QHeaderView::ResizeToContents);
    p_advancedTable->verticalHeader()->setSectionResizeMode(QHeaderView::ResizeToContents);
    p_advancedTable->setVerticalHeaderLabels(channelRowLabels(numChannels));
    p_advancedTable->setSelectionMode(QAbstractItemView::NoSelection);
    p_advancedTable->setFocusPolicy(Qt::NoFocus);

    tabs->addTab(p_standardTable, QString("Standard"));
    tabs->addTab(p_advancedTable, QString("Advanced"));
    leftSide->addWidget(tabs, 1);

    QWidget *lastFocusWidget = nullptr;
    QMetaEnum rt = QMetaEnum::fromType<PulseGenConfig::Role>();

    for(int i=0; i<numChannels; ++i)
    {
        while(static_cast<std::size_t>(i) >= getArraySize(BC::Key::PulseWidget::channels))
            appendArrayMap(BC::Key::PulseWidget::channels, {});

        ChWidgets ch;

        // --- Name (table item; source of truth for the channel name) ---
        auto nameItem = new QTableWidgetItem(cfg.setting(i, PulseGenConfig::NameSetting).toString());
        p_standardTable->setItem(i, StdNameCol, nameItem);

        // --- Role ---
        ch.roleBox = new EnumComboBox<PulseGenConfig::Role>;
        connect(ch.roleBox, qOverload<int>(&QComboBox::currentIndexChanged), this,
                [this, i, rt](int index) {
            auto roleBox = d_widgetList.at(i).roleBox;
            auto r = roleBox->value(index);
            QString name;
            bool nameEditable;
            if(r == PulseGenConfig::None)
            {
                name = QString("Ch%1").arg(i+1);
                nameEditable = true;
            }
            else
            {
                name = QString::fromLatin1(rt.valueToKey(r));
                nameEditable = false;
            }

            applyChannelName(i, name);
            if(auto item = p_standardTable->item(i, StdNameCol))
            {
                auto flags = item->flags();
                if(nameEditable)
                    flags |= Qt::ItemIsEditable;
                else
                    flags &= ~Qt::ItemIsEditable;
                item->setFlags(flags);
            }
            emit changeSetting(d_key, i, PulseGenConfig::NameSetting, name);
            emit changeSetting(d_key, i, PulseGenConfig::RoleSetting, r);
        });
        p_standardTable->setCellWidget(i, StdRoleCol, ch.roleBox);

        // --- Inv? (Active Level checkbox) ---
        ch.invBox = new QCheckBox;
        ch.invBox->setToolTip(QString(
            "Active-low (inverted) output.\n"
            "Unchecked: pulse goes high (default).\n"
            "Checked: pulse goes low."));
        connect(ch.invBox, &QCheckBox::toggled, this, [this, i](bool inv){
            emit changeSetting(d_key, i, PulseGenConfig::LevelSetting,
                               QVariant::fromValue(inv ? PulseGenConfig::ActiveLow
                                                       : PulseGenConfig::ActiveHigh));
        });
        centerCellWidget(p_standardTable, i, StdInvCol, ch.invBox);

        // --- Sync ---
        ch.syncBox = new QComboBox;
        for(int j=0; j<=numChannels; ++j)
        {
            if(j == 0)
                ch.syncBox->addItem(QString("T0"), 0);
            else
                ch.syncBox->addItem(QString("Ch%1").arg(j), j);
        }
        if(auto md = dynamic_cast<QStandardItemModel*>(ch.syncBox->model()))
        {
            if(auto item = dynamic_cast<QStandardItem*>(md->item(i+1)))
                item->setEnabled(false);
        }
        connect(ch.syncBox, qOverload<int>(&QComboBox::currentIndexChanged), this,
                [this, i](int j) {
            auto syncBox = d_widgetList.at(i).syncBox;
            if(ps_config->testCircularSync(i, j))
            {
                QMessageBox::warning(this, QString("Circular Sync"),
                    QString("Cannot set sync channel because of a circular reference "
                            "(i.e., A triggers B, but B triggers A)."),
                    QMessageBox::Ok, QMessageBox::Ok);
                syncBox->blockSignals(true);
                syncBox->setCurrentIndex(ps_config->setting(i, PulseGenConfig::SyncSetting).toInt());
                syncBox->blockSignals(false);
            }
            else
                emit changeSetting(d_key, i, PulseGenConfig::SyncSetting, j);
        });
        ch.syncBox->setEnabled(false);
        ch.syncBox->setToolTip(lockedWhilePulsingTip);
        p_advancedTable->setCellWidget(i, AdvSyncCol, ch.syncBox);

        // --- Delay ---
        ch.delayBox = new QDoubleSpinBox;
        ch.delayBox->setKeyboardTracking(false);
        ch.delayBox->setDecimals(3);
        ch.delayBox->setSuffix(QString::fromUtf16(u" µs"));
        connect(ch.delayBox, vc, this, [this, i](double val){
            emit changeSetting(d_key, i, PulseGenConfig::DelaySetting, val);
        });
        p_standardTable->setCellWidget(i, StdDelayCol, ch.delayBox);

        // --- Width ---
        ch.widthBox = new QDoubleSpinBox;
        ch.widthBox->setKeyboardTracking(false);
        ch.widthBox->setDecimals(3);
        ch.widthBox->setSuffix(QString::fromUtf16(u" µs"));
        connect(ch.widthBox, vc, this, [this, i](double val){
            emit changeSetting(d_key, i, PulseGenConfig::WidthSetting, val);
        });
        p_standardTable->setCellWidget(i, StdWidthCol, ch.widthBox);

        // --- Mode ---
        ch.modeBox = new EnumComboBox<PulseGenConfig::ChannelMode>;
        connect(ch.modeBox, qOverload<int>(&QComboBox::currentIndexChanged), this,
                [this, i](int j) {
            emit changeSetting(d_key, i, PulseGenConfig::ModeSetting,
                               QVariant::fromValue(d_widgetList.at(i).modeBox->value(j)));
        });
        ch.modeBox->setEnabled(false);
        ch.modeBox->setToolTip(lockedWhilePulsingTip);
        p_advancedTable->setCellWidget(i, AdvModeCol, ch.modeBox);

        // --- Enabled ---
        ch.onButton = new QToolButton;
        ch.onButton->setCheckable(true);
        ch.onButton->setAutoRaise(false);
        ch.onButton->setIconSize(QSize(18, 18));
        ch.onButton->setToolTip(QString("Enable or disable this channel."));
        if(canDisable)
        {
            ch.onButton->setChecked(false);
            applyOnIcon(ch.onButton, false);
        }
        else
        {
            ch.onButton->setChecked(true);
            applyOnIcon(ch.onButton, true);
            ch.onButton->setEnabled(false);
        }
        connect(ch.onButton, &QToolButton::toggled, this, [this, i](bool en){
            applyOnIcon(d_widgetList.at(i).onButton, en);
            emit changeSetting(d_key, i, PulseGenConfig::EnabledSetting, en);
        });
        centerCellWidget(p_standardTable, i, StdEnabledCol, ch.onButton);

        // --- Advanced tab cell widgets ---
        ch.dutyOnBox = new QSpinBox;
        ch.dutyOnBox->setMinimum(1);
        connect(ch.dutyOnBox, qOverload<int>(&QSpinBox::valueChanged), this, [this, i](int d){
            emit changeSetting(d_key, i, PulseGenConfig::DutyOnSetting, d);
        });
        p_advancedTable->setCellWidget(i, AdvDutyOnCol, ch.dutyOnBox);

        ch.dutyOffBox = new QSpinBox;
        ch.dutyOffBox->setMinimum(1);
        connect(ch.dutyOffBox, qOverload<int>(&QSpinBox::valueChanged), this, [this, i](int d){
            emit changeSetting(d_key, i, PulseGenConfig::DutyOffSetting, d);
        });
        p_advancedTable->setCellWidget(i, AdvDutyOffCol, ch.dutyOffBox);

        ch.delayStepBox = new QDoubleSpinBox;
        ch.delayStepBox->setDecimals(3);
        ch.delayStepBox->setRange(0.001, 1000.0);
        ch.delayStepBox->setSuffix(QString::fromUtf16(u" µs"));
        connect(ch.delayStepBox, vc, ch.delayBox, &QDoubleSpinBox::setSingleStep);
        p_advancedTable->setCellWidget(i, AdvDelayStepCol, ch.delayStepBox);

        ch.widthStepBox = new QDoubleSpinBox;
        ch.widthStepBox->setDecimals(3);
        ch.widthStepBox->setRange(0.001, 1000.0);
        ch.widthStepBox->setSuffix(QString::fromUtf16(u" µs"));
        connect(ch.widthStepBox, vc, ch.widthBox, &QDoubleSpinBox::setSingleStep);
        p_advancedTable->setCellWidget(i, AdvWidthStepCol, ch.widthStepBox);

        d_widgetList.append(ch);
        lastFocusWidget = ch.onButton;
    }

    // Name-cell direct edits propagate as a NameSetting change. The
    // applyChannelName helper blocks table signals when writing back so
    // there is no feedback loop.
    connect(p_standardTable, &QTableWidget::itemChanged, this, [this](QTableWidgetItem *item){
        if(!item || item->column() != StdNameCol)
            return;
        const int ch = item->row();
        for(int j=0; j<d_widgetList.size(); ++j)
            d_widgetList.at(j).syncBox->setItemText(ch+1, item->text());
        emit changeSetting(d_key, ch, PulseGenConfig::NameSetting, item->text());
    });

    if(lastFocusWidget != nullptr)
        setTabOrder(lastFocusWidget, p_repRateBox);

    updateFromSettings();

    setFocusPolicy(Qt::TabFocus);
    setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);

    // ps_config initialized here; pulse plot constructed after the
    // configuration has been pushed into the widgets so its first draw
    // matches the displayed state.
    p_pulsePlot = nullptr;
    setFromConfig(d_key, cfg);
    p_pulsePlot = new PulsePlot(ps_config, this);
    p_pulsePlot->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    hbl->addWidget(p_pulsePlot, 1);
}

PulseConfigWidget::~PulseConfigWidget()
{
    using namespace BC::Key::PulseWidget;
    for(int i=0; i<d_widgetList.size(); ++i)
    {
        const auto &chw = d_widgetList.at(i);
        setArrayValue(channels, i, delayStep, chw.delayStepBox->value(), false);
        setArrayValue(channels, i, widthStep, chw.widthStepBox->value(), false);
    }
    save();
}

const PulseGenConfig &PulseConfigWidget::getConfig() const
{
    return *ps_config;
}

void PulseConfigWidget::configureForWizard()
{
    d_wizardMode = true;

    connect(this, &PulseConfigWidget::changeSetting, this, &PulseConfigWidget::newSetting);

    for(auto &ch : d_widgetList)
        ch.locked = false;
}

void PulseConfigWidget::applyChannelName(int ch, const QString &name)
{
    if(p_standardTable && ch >= 0 && ch < p_standardTable->rowCount())
    {
        p_standardTable->blockSignals(true);
        if(auto item = p_standardTable->item(ch, StdNameCol))
            item->setText(name);
        p_standardTable->blockSignals(false);
    }
    for(int j=0; j<d_widgetList.size(); ++j)
        d_widgetList.at(j).syncBox->setItemText(ch+1, name);
}

void PulseConfigWidget::newSetting(const QString &key, int index, PulseGenConfig::Setting s, QVariant val)
{
    if(index >= d_widgetList.size() || key != d_key)
        return;

    blockSignals(true);
    ps_config->setCh(index, s, val);

    switch(s) {
    case PulseGenConfig::DelaySetting:
        d_widgetList.at(index).delayBox->setValue(val.toDouble());
        break;
    case PulseGenConfig::WidthSetting:
        d_widgetList.at(index).widthBox->setValue(val.toDouble());
        break;
    case PulseGenConfig::LevelSetting:
        d_widgetList.at(index).invBox->setChecked(
            val.value<PulseGenConfig::ActiveLevel>() == PulseGenConfig::ActiveLow);
        break;
    case PulseGenConfig::EnabledSetting:
        d_widgetList.at(index).onButton->setChecked(val.toBool());
        break;
    case PulseGenConfig::NameSetting:
        applyChannelName(index, val.toString());
        break;
    case PulseGenConfig::RoleSetting:
        d_widgetList.at(index).roleBox->setCurrentValue(val.value<PulseGenConfig::Role>());
        break;
    case PulseGenConfig::ModeSetting:
        d_widgetList.at(index).modeBox->setCurrentValue(val.value<PulseGenConfig::ChannelMode>());
        break;
    case PulseGenConfig::SyncSetting:
        d_widgetList.at(index).syncBox->setCurrentIndex(val.toInt());
        break;
    case PulseGenConfig::DutyOnSetting:
        d_widgetList.at(index).dutyOnBox->setValue(val.toInt());
        break;
    case PulseGenConfig::DutyOffSetting:
        d_widgetList.at(index).dutyOffBox->setValue(val.toInt());
        break;
    case PulseGenConfig::RepRateSetting:
        p_repRateBox->setValue(val.toDouble());
        ps_config->d_repRate = val.toDouble();
        break;
    case PulseGenConfig::PGenModeSetting:
        p_sysModeBox->setCurrentValue(val.value<PulseGenConfig::PGenMode>());
        ps_config->d_mode = val.value<PulseGenConfig::PGenMode>();
        break;
    case PulseGenConfig::PGenEnabledSetting:
        p_sysOnOffButton->setChecked(val.toBool());
        ps_config->d_pulseEnabled = val.toBool();
        break;
    }
    blockSignals(false);

    if(p_pulsePlot)
        p_pulsePlot->updatePulsePlot();
}

void PulseConfigWidget::setFromConfig(const QString &key, const PulseGenConfig &c)
{
    if(key != d_key)
        return;

    blockSignals(true);
    ps_config = std::make_shared<PulseGenConfig>(c);
    for(int i=0; i<qMin(c.size(), (int)d_widgetList.size()); i++)
    {
        const auto &cc = c.d_channels.at(i);
        auto &chw = d_widgetList[i];

        chw.delayBox->setValue(cc.delay);
        chw.widthBox->setValue(cc.width);
        chw.invBox->setChecked(cc.level == PulseGenConfig::ActiveLow);
        chw.onButton->setChecked(cc.enabled);
        chw.modeBox->setCurrentValue(cc.mode);
        chw.roleBox->setCurrentValue(cc.role);
        chw.dutyOnBox->setValue(cc.dutyOn);
        chw.dutyOffBox->setValue(cc.dutyOff);

        chw.syncBox->blockSignals(true);
        chw.syncBox->setCurrentIndex(cc.syncCh);
        chw.syncBox->blockSignals(false);

        applyChannelName(i, cc.channelName);
        if(auto item = p_standardTable->item(i, StdNameCol))
        {
            auto flags = item->flags();
            if(cc.role == PulseGenConfig::None)
                flags |= Qt::ItemIsEditable;
            else
                flags &= ~Qt::ItemIsEditable;
            item->setFlags(flags);
        }

        chw.modeBox->setDisabled(c.d_pulseEnabled || chw.locked);
        chw.syncBox->setDisabled(c.d_pulseEnabled || chw.locked);
    }
    p_repRateBox->setValue(c.d_repRate);
    p_repRateBox->setEnabled(c.d_mode == PulseGenConfig::Continuous);
    p_sysModeBox->setCurrentValue(c.d_mode);
    p_sysModeBox->setDisabled(c.d_pulseEnabled);
    p_sysOnOffButton->setChecked(c.d_pulseEnabled);
    p_sysOnOffButton->setText(c.d_pulseEnabled ? QString("On") : QString("Off"));
    blockSignals(false);

    if(p_pulsePlot)
        p_pulsePlot->newConfig(ps_config);
}

void PulseConfigWidget::updateFromSettings()
{
    using namespace BC::Key::PulseWidget;
    SettingsStorage s(d_key, Hardware);
    for(int i=0; i<d_widgetList.size(); i++)
    {
        auto &chw = d_widgetList[i];

        const double dStep = getArrayValue(channels, i, delayStep, 1.0);
        const double wStep = getArrayValue(channels, i, widthStep, 1.0);

        chw.delayStepBox->setValue(dStep);
        chw.widthStepBox->setValue(wStep);

        chw.delayBox->blockSignals(true);
        chw.delayBox->setRange(s.get<double>(BC::Key::PGen::minDelay, 0.0),
                               s.get<double>(BC::Key::PGen::maxDelay, 1e5));
        chw.delayBox->setSingleStep(dStep);
        chw.delayBox->blockSignals(false);

        chw.widthBox->blockSignals(true);
        chw.widthBox->setRange(s.get<double>(BC::Key::PGen::minWidth, 0.010),
                               s.get<double>(BC::Key::PGen::maxWidth, 1e5));
        chw.widthBox->setSingleStep(wStep);
        chw.widthBox->blockSignals(false);

        auto r = ps_config->setting(i, PulseGenConfig::RoleSetting).value<PulseGenConfig::Role>();
        chw.roleBox->setCurrentValue(r);

        auto n = ps_config->setting(i, PulseGenConfig::NameSetting).toString();
        applyChannelName(i, n);
    }

    p_repRateBox->setRange(s.get(BC::Key::PGen::minRepRate, 0.01),
                           s.get(BC::Key::PGen::maxRepRate, 1e5));
}

QSize PulseConfigWidget::sizeHint() const
{
    if(!p_standardTable || !p_pulsePlot)
        return QWidget::sizeHint();

    // Per-row height: estimate from a sample cell-widget hint, fall back
    // to a reasonable value for combobox/spinbox rows.
    int rowH = 32;
    if(!d_widgetList.isEmpty() && d_widgetList.first().delayBox)
        rowH = std::max(rowH, d_widgetList.first().delayBox->sizeHint().height() + 6);

    const int numRows = p_standardTable->rowCount();
    int hdrH = p_standardTable->horizontalHeader()->sizeHint().height();
    if(hdrH <= 0) hdrH = 26;
    const int tableH = hdrH + numRows * rowH + p_standardTable->frameWidth() * 2 + 4;

    // Width: vertical-header + sum of per-column widths derived from
    // cell-widget and section-header hints, padded for QTableWidget's
    // internal cell margins. The Name column has a generous floor so the
    // stretch column opens with comfortable room for channel labels.
    constexpr int cellPadding = 18;
    int tableW = p_standardTable->verticalHeader()->sizeHint().width();
    if(tableW <= 0) tableW = 32;
    for(int c=0; c<p_standardTable->columnCount(); ++c)
    {
        int cw = 0;
        if(auto w = p_standardTable->cellWidget(0, c))
            cw = w->sizeHint().width() + cellPadding;
        if(cw <= 0)
            cw = 80;
        const int hdrW = p_standardTable->horizontalHeader()->sectionSizeHint(c);
        cw = std::max(cw, hdrW);
        if(c == StdNameCol)
            cw = std::max(cw, 240);
        tableW += cw;
    }
    tableW += p_standardTable->frameWidth() * 2;

    const QSize sysHint = p_mainGb ? p_mainGb->sizeHint() : QSize(320, 90);
    const int leftH = sysHint.height() + 36 + tableH;  // sys + tab chrome + table
    const int leftW = std::max(tableW, sysHint.width());

    const QSize plotHint = p_pulsePlot->sizeHint();
    return {leftW + plotHint.width() + 12, std::max(leftH, plotHint.height())};
}
