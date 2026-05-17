#include <gui/widget/ftmwacquisitionpanel.h>

#include <climits>

#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QSpinBox>
#include <QPushButton>

#include <gui/style/themecolors.h>
#include <gui/widget/settingstable.h>

using namespace Qt::StringLiterals;

FtmwAcquisitionPanel::FtmwAcquisitionPanel(bool main, QWidget *parent) :
    QWidget(parent), d_main(main)
{
    auto *outer = new QVBoxLayout;
    outer->setContentsMargins(4,4,4,4);

    auto *table = new SettingsTable(this);
    table->setFocusPolicy(Qt::NoFocus);

    p_refreshBox = new QSpinBox;
    p_refreshBox->setRange(500,60000);
    p_refreshBox->setSingleStep(500);
    p_refreshBox->setSuffix(" ms"_L1);
    p_refreshBox->setAlignment(Qt::AlignCenter);
    p_refreshBox->setKeyboardTracking(false);
    p_refreshBox->setEnabled(false);
    {
        const auto tip = "How often the live FID and FT plots refresh during acquisition."_L1;
        p_refreshBox->setToolTip(tip);
        int refreshRow = table->addSettingRow("Refresh Interval"_L1, p_refreshBox, tip);
        if(!d_main)
            table->setRowHidden(refreshRow, true);
    }

    p_averagesBox = new QSpinBox;
    p_averagesBox->setRange(1,INT_MAX);
    p_averagesBox->setSingleStep(25);
    p_averagesBox->setAlignment(Qt::AlignCenter);
    p_averagesBox->setKeyboardTracking(false);
    p_averagesBox->setEnabled(false);
    p_averagesBox->setToolTip("Number of shots in the rolling (peak-up) average shown on the live plots."_L1);
    table->addSettingRow("Peak Up Averages"_L1, p_averagesBox,
                         "Number of shots in the rolling (peak-up) average shown on the live plots."_L1);

    outer->addWidget(table,0);

    // Reset / Backup share one row with short labels; the tooltips
    // carry the full meaning. Keeps the panel as compact as its two
    // settings rows.
    p_resetAveragesButton = new QPushButton(
        ThemeColors::createThemedIcon(":/icons/arrow-path.svg",ThemeColors::IconSecondary,this),
        "Reset"_L1);
    p_resetAveragesButton->setToolTip("Reset the rolling average (peak-up) accumulation."_L1);
    p_resetAveragesButton->setEnabled(false);

    p_manualBackupButton = new QPushButton(
        ThemeColors::createThemedIcon(":/icons/archive-box-arrow-down.svg",ThemeColors::IconSecondary,this),
        "Backup"_L1);
    p_manualBackupButton->setToolTip("Save a backup snapshot of the current FID list."_L1);
    p_manualBackupButton->setEnabled(false);
    p_manualBackupButton->setVisible(d_main);

    auto *btnRow = new QHBoxLayout;
    btnRow->setContentsMargins(0,0,0,0);
    btnRow->addWidget(p_resetAveragesButton);
    btnRow->addWidget(p_manualBackupButton);
    outer->addLayout(btnRow,0);

    setLayout(outer);

    connect(p_refreshBox, qOverload<int>(&QSpinBox::valueChanged), this,
            &FtmwAcquisitionPanel::refreshIntervalChanged);
    connect(p_averagesBox, qOverload<int>(&QSpinBox::valueChanged), this,
            &FtmwAcquisitionPanel::averagesChanged);
    connect(p_resetAveragesButton, &QPushButton::clicked, this,
            &FtmwAcquisitionPanel::resetAveragesClicked);
    connect(p_manualBackupButton, &QPushButton::clicked, this, [this](){
        // Disable immediately so a single click cannot stack requests; the
        // owner re-enables once the write completes.
        p_manualBackupButton->setEnabled(false);
        emit manualBackupClicked();
    });
}

int FtmwAcquisitionPanel::refreshInterval() const
{
    return p_refreshBox->value();
}

void FtmwAcquisitionPanel::setRefreshInterval(int ms)
{
    p_refreshBox->blockSignals(true);
    p_refreshBox->setValue(ms);
    p_refreshBox->blockSignals(false);
}

void FtmwAcquisitionPanel::setRefreshEnabled(bool enabled)
{
    p_refreshBox->setEnabled(enabled);
}

void FtmwAcquisitionPanel::setAverages(int n)
{
    p_averagesBox->blockSignals(true);
    p_averagesBox->setValue(n);
    p_averagesBox->blockSignals(false);
}

void FtmwAcquisitionPanel::setPeakUpControlsEnabled(bool enabled)
{
    p_averagesBox->setEnabled(enabled);
    p_resetAveragesButton->setEnabled(enabled);
}

void FtmwAcquisitionPanel::setManualBackupEnabled(bool enabled)
{
    if(d_main)
        p_manualBackupButton->setEnabled(enabled);
}
