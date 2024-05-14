#include "ftmwplottoolbar.h"

#include <QToolButton>
#include <QMenu>

#include <gui/widget/toolbarwidgetaction.h>

FtmwPlotToolBar::FtmwPlotToolBar(QWidget *parent) : QToolBar(parent)
{
    qRegisterMetaType<FtmwPlotToolBar::MainPlotMode>();

    addAction(new LabelWidgetAction("Main Plot"));

    p_mainPlotBox = new EnumComboBoxWidgetAction<MainPlotMode>("Mode",this);
    p_mainPlotBox->setCurrentValue(Live);
    p_mainPlotBox->setToolTip("Select what is displayed on the main (large) FT Plot");
    connect(p_mainPlotBox,&EnumComboBoxWidgetAction<MainPlotMode>::valueChanged,this,&FtmwPlotToolBar::mainPlotSettingChanged);
    addAction(p_mainPlotBox);

    QMenu *procMenu = new QMenu;
    auto ptb = new QToolButton(this);
    ptb->setText("Sideband Processing");
    ptb->setToolTip("Configure options for sideband deconvolution.");
    ptb->setMenu(procMenu);
    ptb->setPopupMode(QToolButton::InstantPopup);
    p_sbProcAction = addWidget(ptb);

    p_frameBox = new SpinBoxWidgetAction("Frame",this);
    p_frameBox->setToolTip("Select which frame is used in sideband deconvolution.");
    p_frameBox->setRange(1,1);
    p_frameBox->setValue(1);
    p_frameBox->setSpecialValueText("Average");
    connect(p_frameBox,&SpinBoxWidgetAction::valueChanged,this,&FtmwPlotToolBar::mainPlotSettingChanged);
    procMenu->addAction(p_frameBox);

    p_sbMinBox = new DoubleSpinBoxWidgetAction("Min Offset",this);
    p_sbMinBox->setRange(0,1000000.0);
    p_sbMinBox->setValue(0.0);
    p_sbMinBox->setDecimals(3);
    p_sbMinBox->setSuffix(" MHz");
    p_sbMinBox->setToolTip("Minimum offset frequency included in sideband deconvolution algorithm.");
    connect(p_sbMinBox,&DoubleSpinBoxWidgetAction::valueChanged,this,&FtmwPlotToolBar::mainPlotSettingChanged);
    procMenu->addAction(p_sbMinBox);

    p_sbMaxBox = new DoubleSpinBoxWidgetAction("Max Offset",this);
    p_sbMaxBox->setRange(0,1000000.0);
    p_sbMaxBox->setValue(100.0);
    p_sbMaxBox->setDecimals(3);
    p_sbMaxBox->setSuffix(" MHz");
    p_sbMaxBox->setToolTip("Maximum offset frequency included in sideband deconvolution algorithm.");
    connect(p_sbMaxBox,&DoubleSpinBoxWidgetAction::valueChanged,this,&FtmwPlotToolBar::mainPlotSettingChanged);
    procMenu->addAction(p_sbMaxBox);

    p_sbAlgoBox = new EnumComboBoxWidgetAction<FtWorker::DeconvolutionMethod>("Avg Algorithm",this);
    p_sbAlgoBox->setCurrentValue(FtWorker::Harmonic_Mean);
    p_sbAlgoBox->setToolTip("Averaging algorithm to suppress signals in undesired sideband.");
    connect(p_sbAlgoBox,&EnumComboBoxWidgetAction<FtWorker::DeconvolutionMethod>::valueChanged,this,&FtmwPlotToolBar::mainPlotSettingChanged);
    procMenu->addAction(p_sbAlgoBox);

    p_sbProcAction->setVisible(false);

    auto rpAct = procMenu->addAction("Reprocess");
    connect(rpAct,&QAction::triggered,this,&FtmwPlotToolBar::mainPlotSettingChanged);

    for(int i=1; i<3; ++i)
    {

        addSeparator();

        addAction(new LabelWidgetAction(QString("Plot %1").arg(i)));

        auto sb = new SpinBoxWidgetAction("Segment",this);
        sb->setRange(1,1);
        connect(sb,&SpinBoxWidgetAction::valueChanged,[this,i](){ emit plotSettingChanged(i); });
        d_seg.insert({i,sb});
        addAction(sb);

        auto fb = new SpinBoxWidgetAction("Frame",this);
        fb->setRange(1,1);
        fb->setSpecialValueText(QString("Average"));
        connect(fb,&SpinBoxWidgetAction::valueChanged,[this,i](){ emit plotSettingChanged(i); });
        d_frame.insert({i,fb});
        addAction(fb);

        auto bb = new SpinBoxWidgetAction("Backup",this);
        bb->setSpecialValueText("All");
        bb->setRange(0,0);
        connect(bb,&SpinBoxWidgetAction::valueChanged,[this,i](){ emit plotSettingChanged(i); });
        d_backup.insert({i,bb});
        addAction(bb);
    }



}

void FtmwPlotToolBar::prepareForExperiment(const Experiment &e)
{
    if(e.ftmwEnabled())
    {
        p_mainPlotBox->setItemEnabled(Live,true);
        p_mainPlotBox->setCurrentValue(Live);


        if(e.ftmwConfig()->d_type == FtmwConfig::LO_Scan)
        {
            p_mainPlotBox->setItemEnabled(Both_SideBands,true);
            p_mainPlotBox->setItemEnabled(Upper_SideBand,true);
            p_mainPlotBox->setItemEnabled(Lower_SideBand,true);
            p_sbProcAction->setEnabled(true);
            p_sbProcAction->setVisible(true);
        }
        else
        {
            p_mainPlotBox->setItemEnabled(Both_SideBands,false);
            p_mainPlotBox->setItemEnabled(Upper_SideBand,false);
            p_mainPlotBox->setItemEnabled(Lower_SideBand,false);
            p_sbProcAction->setEnabled(false);
            p_sbProcAction->setVisible(false);
        }

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

        p_frameBox->blockSignals(true);
        p_frameBox->setRange(1,e.ftmwConfig()->scopeConfig().d_numRecords);
        if(e.ftmwConfig()->scopeConfig().d_numRecords > 1)
            p_frameBox->setMinimum(0);
        p_frameBox->blockSignals(false);

        for(auto it = d_seg.begin(); it != d_seg.end(); ++it)
        {
            it->second->blockSignals(true);
            it->second->setRange(1,e.ftmwConfig()->d_rfConfig.numSegments());
            it->second->blockSignals(false);
        }
        for(auto it = d_frame.begin(); it != d_frame.end(); ++it)
        {
            it->second->blockSignals(true);
            it->second->setRange(1,e.ftmwConfig()->scopeConfig().d_numRecords);
            if(e.ftmwConfig()->scopeConfig().d_numRecords > 1)
                it->second->setMinimum(0);
            it->second->blockSignals(false);
        }
        for(auto it = d_backup.begin(); it != d_backup.end(); ++it)
        {
            it->second->blockSignals(true);
            it->second->setRange(0,0);
            it->second->setEnabled(false);
            it->second->blockSignals(false);
        }

    }


    setEnabled(e.ftmwEnabled());
}

void FtmwPlotToolBar::experimentComplete()
{
    if(mainPlotMode() == Live)
        p_mainPlotBox->setCurrentValue(FT1);

    p_mainPlotBox->setItemEnabled(Live,false);
}

void FtmwPlotToolBar::newBackup(int n)
{
    for(auto &[key,box] : d_backup)
    {
        Q_UNUSED(key)
        box->setMaximum(n);
        box->setEnabled(true);
    }
}

FtmwPlotToolBar::MainPlotMode FtmwPlotToolBar::mainPlotMode() const
{
    return p_mainPlotBox->value();
}

int FtmwPlotToolBar::sbFrame() const
{
    return p_frameBox->value();
}

double FtmwPlotToolBar::sbMinFreq() const
{
    return p_sbMinBox->value();
}

double FtmwPlotToolBar::sbMaxFreq() const
{
    return p_sbMaxBox->value();
}

FtWorker::DeconvolutionMethod FtmwPlotToolBar::dcMethod() const
{
    return p_sbAlgoBox->value();
}

int FtmwPlotToolBar::frame(int id) const
{
    auto it = d_frame.find(id);
    if(it != d_frame.end())
        return it->second->value();

    return 0;
}

int FtmwPlotToolBar::segment(int id) const
{
    auto it = d_seg.find(id);
    if(it != d_seg.end())
        return it->second->value();

    return 0;
}

int FtmwPlotToolBar::backup(int id) const
{
    auto it = d_backup.find(id);
    if(it != d_backup.end())
        return it->second->value();

    return 0;
}

bool FtmwPlotToolBar::viewingBackup(int plotId) const
{
    return backup(plotId) > 0;
}
