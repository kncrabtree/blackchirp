#include <gui/widget/ftmwacquisitionpanel.h>

#include <climits>

#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QTableWidget>
#include <QHeaderView>
#include <QSpinBox>
#include <QPushButton>

#include <gui/style/themecolors.h>
#include <gui/widget/tablefit.h>

FtmwAcquisitionPanel::FtmwAcquisitionPanel(bool main, QWidget *parent) :
    QWidget(parent), d_main(main)
{
    auto *outer = new QVBoxLayout;
    outer->setContentsMargins(4,4,4,4);

    auto *table = new QTableWidget(2,1,this);
    table->setVerticalHeaderLabels({"Refresh Interval","Peak Up Averages"});
    table->horizontalHeader()->setVisible(false);
    table->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
    table->verticalHeader()->setSectionResizeMode(QHeaderView::ResizeToContents);
    table->setSelectionMode(QAbstractItemView::NoSelection);
    table->setFocusPolicy(Qt::NoFocus);
    table->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOn);
    table->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);

    p_refreshBox = new QSpinBox;
    p_refreshBox->setRange(500,60000);
    p_refreshBox->setSingleStep(500);
    p_refreshBox->setSuffix(" ms");
    p_refreshBox->setAlignment(Qt::AlignCenter);
    p_refreshBox->setKeyboardTracking(false);
    p_refreshBox->setEnabled(false);
    table->setCellWidget(0,0,p_refreshBox);

    p_averagesBox = new QSpinBox;
    p_averagesBox->setRange(1,INT_MAX);
    p_averagesBox->setSingleStep(25);
    p_averagesBox->setAlignment(Qt::AlignCenter);
    p_averagesBox->setKeyboardTracking(false);
    p_averagesBox->setEnabled(false);
    table->setCellWidget(1,0,p_averagesBox);

    if(!d_main)
        table->hideRow(0);

    fitTableToContents(table);
    outer->addWidget(table,0);

    // Reset / Backup share one row with short labels; the tooltips
    // carry the full meaning. Keeps the panel as compact as its two
    // settings rows.
    p_resetAveragesButton = new QPushButton(
        ThemeColors::createThemedIcon(":/icons/arrow-path.svg",ThemeColors::IconSecondary,this),
        "Reset");
    p_resetAveragesButton->setToolTip("Reset the rolling average (peak-up) accumulation.");
    p_resetAveragesButton->setEnabled(false);

    p_manualBackupButton = new QPushButton(
        ThemeColors::createThemedIcon(":/icons/archive-box-arrow-down.svg",ThemeColors::IconSecondary,this),
        "Backup");
    p_manualBackupButton->setToolTip("Save a backup snapshot of the current FID list.");
    p_manualBackupButton->setEnabled(false);
    p_manualBackupButton->setVisible(d_main);

    auto *btnRow = new QHBoxLayout;
    btnRow->setContentsMargins(0,0,0,0);
    btnRow->addWidget(p_resetAveragesButton);
    btnRow->addWidget(p_manualBackupButton);
    outer->addLayout(btnRow,0);

    outer->addStretch(1);
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
