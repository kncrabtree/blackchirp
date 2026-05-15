#include <gui/widget/ftmwplotpanel.h>

#include <QVBoxLayout>
#include <QTableWidget>
#include <QHeaderView>
#include <QSpinBox>
#include <QDoubleSpinBox>
#include <QComboBox>
#include <QCheckBox>
#include <QPushButton>
#include <QLineEdit>
#include <QStandardItemModel>
#include <QMetaEnum>

#include <gui/widget/cellwidgethelpers.h>

using BC::Gui::centerCellWidget;

namespace {

QComboBox *makeCenteredCombo()
{
    auto *cb = new QComboBox;
    cb->setEditable(true);
    cb->lineEdit()->setReadOnly(true);
    cb->lineEdit()->setAlignment(Qt::AlignCenter);
    return cb;
}

void recenterCombo(QComboBox *cb)
{
    for(int i=0; i<cb->count(); ++i)
        cb->setItemData(i,Qt::AlignCenter,Qt::TextAlignmentRole);
}

}

FtmwPlotPanel::FtmwPlotPanel(QWidget *parent) : QWidget(parent)
{
    qRegisterMetaType<FtmwPlotPanel::MainPlotMode>();

    auto *outer = new QVBoxLayout;
    outer->setContentsMargins(4,4,4,4);

    p_table = new QTableWidget(d_rowCount,1,this);
    p_table->setVerticalHeaderLabels({
        "Main Plot Mode",
        "SB Frame",
        "SB Min Offset",
        "SB Max Offset",
        "SB Avg Algorithm",
        "Plot 1 Segment",
        "Plot 1 Frame",
        "Plot 1 Backup",
        "Plot 1 Differential",
        "Plot 2 Segment",
        "Plot 2 Frame",
        "Plot 2 Backup",
        "Plot 2 Differential"});
    p_table->horizontalHeader()->setVisible(false);
    p_table->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
    p_table->verticalHeader()->setSectionResizeMode(QHeaderView::ResizeToContents);
    p_table->setSelectionMode(QAbstractItemView::NoSelection);
    p_table->setFocusPolicy(Qt::NoFocus);
    p_table->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOn);
    p_table->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);

    // ─── Main Plot Mode ────────────────────────────────────────────────
    p_mainPlotBox = makeCenteredCombo();
    {
        auto me = QMetaEnum::fromType<MainPlotMode>();
        for(int i=0; i<me.keyCount(); ++i)
            p_mainPlotBox->addItem(QString::fromLatin1(me.key(i)),
                                   QVariant::fromValue<MainPlotMode>(static_cast<MainPlotMode>(me.value(i))));
    }
    recenterCombo(p_mainPlotBox);
    p_mainPlotBox->setCurrentIndex(p_mainPlotBox->findData(QVariant::fromValue(Live)));
    p_mainPlotBox->setToolTip("Select what is displayed on the main (large) FT Plot");
    p_table->setCellWidget(d_rowMainMode,0,p_mainPlotBox);
    connect(p_mainPlotBox, qOverload<int>(&QComboBox::currentIndexChanged),
            this, [this](int){ emit mainPlotSettingChanged(); });

    // ─── Sideband Processing rows (visible for LO_Scan only) ───────────
    p_sbFrameBox = new QSpinBox;
    p_sbFrameBox->setRange(1,1);
    p_sbFrameBox->setValue(1);
    p_sbFrameBox->setSpecialValueText("Average");
    p_sbFrameBox->setAlignment(Qt::AlignCenter);
    p_sbFrameBox->setKeyboardTracking(false);
    p_sbFrameBox->setToolTip("Select which frame is used in sideband deconvolution.");
    p_table->setCellWidget(d_rowSbFrame,0,p_sbFrameBox);

    p_sbMinBox = new QDoubleSpinBox;
    p_sbMinBox->setRange(0,1000000.0);
    p_sbMinBox->setValue(0.0);
    p_sbMinBox->setDecimals(3);
    p_sbMinBox->setSuffix(" MHz");
    p_sbMinBox->setAlignment(Qt::AlignCenter);
    p_sbMinBox->setKeyboardTracking(false);
    p_sbMinBox->setToolTip("Minimum offset frequency included in sideband deconvolution algorithm.");
    p_table->setCellWidget(d_rowSbMin,0,p_sbMinBox);

    p_sbMaxBox = new QDoubleSpinBox;
    p_sbMaxBox->setRange(0,1000000.0);
    p_sbMaxBox->setValue(100.0);
    p_sbMaxBox->setDecimals(3);
    p_sbMaxBox->setSuffix(" MHz");
    p_sbMaxBox->setAlignment(Qt::AlignCenter);
    p_sbMaxBox->setKeyboardTracking(false);
    p_sbMaxBox->setToolTip("Maximum offset frequency included in sideband deconvolution algorithm.");
    p_table->setCellWidget(d_rowSbMax,0,p_sbMaxBox);

    p_sbAlgoBox = makeCenteredCombo();
    {
        auto me = QMetaEnum::fromType<FtWorker::DeconvolutionMethod>();
        for(int i=0; i<me.keyCount(); ++i)
            p_sbAlgoBox->addItem(QString::fromLatin1(me.key(i)),
                                 QVariant::fromValue<FtWorker::DeconvolutionMethod>(static_cast<FtWorker::DeconvolutionMethod>(me.value(i))));
    }
    recenterCombo(p_sbAlgoBox);
    p_sbAlgoBox->setCurrentIndex(p_sbAlgoBox->findData(QVariant::fromValue(FtWorker::Harmonic_Mean)));
    p_sbAlgoBox->setToolTip("Averaging algorithm to suppress signals in undesired sideband.");
    p_table->setCellWidget(d_rowSbAlgo,0,p_sbAlgoBox);

    connect(p_sbFrameBox, qOverload<int>(&QSpinBox::valueChanged), this,
            [this](int){ emit mainPlotSettingChanged(); });
    connect(p_sbMinBox, qOverload<double>(&QDoubleSpinBox::valueChanged), this,
            [this](double){ emit mainPlotSettingChanged(); });
    connect(p_sbMaxBox, qOverload<double>(&QDoubleSpinBox::valueChanged), this,
            [this](double){ emit mainPlotSettingChanged(); });
    connect(p_sbAlgoBox, qOverload<int>(&QComboBox::currentIndexChanged), this,
            [this](int){ emit mainPlotSettingChanged(); });

    // ─── Per-plot rows (Plot 1, Plot 2) ─────────────────────────────────
    buildPlotControls(1, d_rowPlot1Seg, d_rowPlot1Frame, d_rowPlot1Backup, d_rowPlot1Diff);
    buildPlotControls(2, d_rowPlot2Seg, d_rowPlot2Frame, d_rowPlot2Backup, d_rowPlot2Diff);

    outer->addWidget(p_table,1);

    p_sbReprocessButton = new QPushButton("Reprocess Sidebands");
    p_sbReprocessButton->setToolTip("Re-run sideband deconvolution with the current settings.");
    outer->addWidget(p_sbReprocessButton,0);
    connect(p_sbReprocessButton, &QPushButton::clicked, this,
            [this](){ emit mainPlotSettingChanged(); });

    setSidebandRowsVisible(false);

    setLayout(outer);
}

void FtmwPlotPanel::buildPlotControls(int id, int segRow, int frameRow, int backupRow, int diffRow)
{
    PlotControls pc;
    pc.seg = new QSpinBox;
    pc.seg->setRange(1,1);
    pc.seg->setAlignment(Qt::AlignCenter);
    pc.seg->setKeyboardTracking(false);
    p_table->setCellWidget(segRow,0,pc.seg);

    pc.frame = new QSpinBox;
    pc.frame->setRange(1,1);
    pc.frame->setSpecialValueText("Average");
    pc.frame->setAlignment(Qt::AlignCenter);
    pc.frame->setKeyboardTracking(false);
    p_table->setCellWidget(frameRow,0,pc.frame);

    pc.backup = new QSpinBox;
    pc.backup->setRange(0,0);
    pc.backup->setSpecialValueText("All");
    pc.backup->setAlignment(Qt::AlignCenter);
    pc.backup->setKeyboardTracking(false);
    p_table->setCellWidget(backupRow,0,pc.backup);

    pc.differential = new QCheckBox;
    pc.differential->setToolTip("If checked, display all shots recorded since the indicated backup.");
    centerCellWidget(p_table,diffRow,0,pc.differential);

    connect(pc.seg, qOverload<int>(&QSpinBox::valueChanged), this,
            [this,id](int){ emit plotSettingChanged(id); });
    connect(pc.frame, qOverload<int>(&QSpinBox::valueChanged), this,
            [this,id](int){ emit plotSettingChanged(id); });
    connect(pc.backup, qOverload<int>(&QSpinBox::valueChanged), this,
            [this,id](int){ emit plotSettingChanged(id); });
    connect(pc.differential, &QCheckBox::toggled, this,
            [this,id](bool){ emit plotSettingChanged(id); });

    d_plotControls.insert({id,pc});
}

void FtmwPlotPanel::setSidebandRowsVisible(bool visible)
{
    for(int row : {d_rowSbFrame, d_rowSbMin, d_rowSbMax, d_rowSbAlgo})
        p_table->setRowHidden(row, !visible);
    p_sbReprocessButton->setVisible(visible);
}

void FtmwPlotPanel::setMainPlotItemEnabled(MainPlotMode mode, bool enabled)
{
    auto *model = qobject_cast<QStandardItemModel*>(p_mainPlotBox->model());
    if(!model) return;
    int idx = p_mainPlotBox->findData(QVariant::fromValue(mode));
    if(idx < 0) return;
    auto *item = model->item(idx);
    if(!item) return;
    auto flags = item->flags();
    if(enabled)
        flags |= Qt::ItemIsEnabled;
    else
        flags &= ~Qt::ItemIsEnabled;
    item->setFlags(flags);
}

void FtmwPlotPanel::prepareForExperiment(const Experiment &e)
{
    if(e.ftmwEnabled())
    {
        setMainPlotItemEnabled(Live,true);
        p_mainPlotBox->setCurrentIndex(p_mainPlotBox->findData(QVariant::fromValue(Live)));

        const bool isLoScan = (e.ftmwConfig()->d_type == FtmwConfig::LO_Scan);
        setMainPlotItemEnabled(Both_SideBands,isLoScan);
        setMainPlotItemEnabled(Upper_SideBand,isLoScan);
        setMainPlotItemEnabled(Lower_SideBand,isLoScan);
        setSidebandRowsVisible(isLoScan);

        auto chirpOffsetRange = e.ftmwConfig()->d_rfConfig.calculateChirpAbsOffsetRange();
        if(chirpOffsetRange.first < 0.0)
            chirpOffsetRange.first = 0.0;
        if(chirpOffsetRange.second < 0.0)
            chirpOffsetRange.second = e.ftmwConfig()->ftNyquistMHz();

        p_sbMinBox->blockSignals(true);
        p_sbMinBox->setRange(0.0,e.ftmwConfig()->ftNyquistMHz());
        p_sbMinBox->setValue(chirpOffsetRange.first);
        p_sbMinBox->blockSignals(false);

        p_sbMaxBox->blockSignals(true);
        p_sbMaxBox->setRange(0.0,e.ftmwConfig()->ftNyquistMHz());
        p_sbMaxBox->setValue(chirpOffsetRange.second);
        p_sbMaxBox->blockSignals(false);

        p_sbFrameBox->blockSignals(true);
        p_sbFrameBox->setRange(1,e.ftmwConfig()->digitizerConfig().d_numRecords);
        if(e.ftmwConfig()->digitizerConfig().d_numRecords > 1)
            p_sbFrameBox->setMinimum(0);
        p_sbFrameBox->blockSignals(false);

        for(auto &[key,pc] : d_plotControls)
        {
            Q_UNUSED(key)
            pc.seg->blockSignals(true);
            pc.seg->setRange(1,e.ftmwConfig()->d_rfConfig.numSegments());
            pc.seg->blockSignals(false);

            pc.frame->blockSignals(true);
            pc.frame->setRange(1,e.ftmwConfig()->digitizerConfig().d_numRecords);
            if(e.ftmwConfig()->digitizerConfig().d_numRecords > 1)
                pc.frame->setMinimum(0);
            pc.frame->blockSignals(false);

            pc.backup->blockSignals(true);
            pc.backup->setRange(0,0);
            pc.backup->setEnabled(false);
            pc.backup->blockSignals(false);

            pc.differential->blockSignals(true);
            pc.differential->setEnabled(false);
            pc.differential->setChecked(false);
            pc.differential->blockSignals(false);
        }
    }

    setEnabled(e.ftmwEnabled());
}

void FtmwPlotPanel::experimentComplete()
{
    if(mainPlotMode() == Live)
        p_mainPlotBox->setCurrentIndex(p_mainPlotBox->findData(QVariant::fromValue(FT1)));

    setMainPlotItemEnabled(Live,false);
}

void FtmwPlotPanel::newBackup(int n)
{
    for(auto &[key,pc] : d_plotControls)
    {
        Q_UNUSED(key)
        pc.backup->setMaximum(n);
        pc.backup->setEnabled(true);
        pc.differential->setEnabled(true);
    }
}

FtmwPlotPanel::MainPlotMode FtmwPlotPanel::mainPlotMode() const
{
    return p_mainPlotBox->currentData().value<MainPlotMode>();
}

int FtmwPlotPanel::sbFrame() const { return p_sbFrameBox->value(); }
double FtmwPlotPanel::sbMinFreq() const { return p_sbMinBox->value(); }
double FtmwPlotPanel::sbMaxFreq() const { return p_sbMaxBox->value(); }
FtWorker::DeconvolutionMethod FtmwPlotPanel::dcMethod() const
{
    return p_sbAlgoBox->currentData().value<FtWorker::DeconvolutionMethod>();
}

int FtmwPlotPanel::frame(int id) const
{
    auto it = d_plotControls.find(id);
    return (it != d_plotControls.end()) ? it->second.frame->value() : 0;
}

int FtmwPlotPanel::segment(int id) const
{
    auto it = d_plotControls.find(id);
    return (it != d_plotControls.end()) ? it->second.seg->value() : 0;
}

int FtmwPlotPanel::backup(int id) const
{
    auto it = d_plotControls.find(id);
    return (it != d_plotControls.end()) ? it->second.backup->value() : 0;
}

bool FtmwPlotPanel::differential(int id) const
{
    auto it = d_plotControls.find(id);
    return (it != d_plotControls.end()) ? it->second.differential->isChecked() : false;
}

bool FtmwPlotPanel::viewingBackup(int plotId) const { return backup(plotId) > 0; }
