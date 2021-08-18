#include "ftmwplottoolbar.h"

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

    p_followBox = new SpinBoxWidgetAction("Follow",this);
    p_followBox->setRange(1,2);
    p_followBox->setPrefix("Plot ");
    p_followBox->setToolTip("Use frame/segment/backup settings from indicated plot (only applicable in some main plot modes)");
    p_followBox->setValue(1);
    connect(p_followBox,&SpinBoxWidgetAction::valueChanged,this,&FtmwPlotToolBar::mainPlotSettingChanged);
    addAction(p_followBox);

    p_sbMinBox = new DoubleSpinBoxWidgetAction("Min Offset",this);
    p_sbMinBox->setRange(0,1000000.0);
    p_sbMinBox->setValue(0.0);
    p_sbMinBox->setDecimals(3);
    p_sbMinBox->setSuffix(" MHz");
    p_sbMinBox->setToolTip("Minimum offset frequency included in sideband deconvolution algorithm.");
    connect(p_sbMinBox,&DoubleSpinBoxWidgetAction::valueChanged,this,&FtmwPlotToolBar::mainPlotSettingChanged);
    addAction(p_sbMinBox);

    p_sbMaxBox = new DoubleSpinBoxWidgetAction("Max Offset",this);
    p_sbMaxBox->setRange(0,1000000.0);
    p_sbMaxBox->setValue(100.0);
    p_sbMaxBox->setDecimals(3);
    p_sbMaxBox->setSuffix(" MHz");
    p_sbMaxBox->setToolTip("Maximum offset frequency included in sideband deconvolution algorithm.");
    connect(p_sbMaxBox,&DoubleSpinBoxWidgetAction::valueChanged,this,&FtmwPlotToolBar::mainPlotSettingChanged);
    addAction(p_sbMaxBox);

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
        connect(fb,&SpinBoxWidgetAction::valueChanged,[this,i](){ emit plotSettingChanged(i); });
        addAction(fb);

        auto bb = new SpinBoxWidgetAction("Backup",this);
        bb->setSpecialValueText("All");
        bb->setRange(0,0);
        connect(bb,&SpinBoxWidgetAction::valueChanged,[this,i](){ emit plotSettingChanged(i); });
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
            p_sbMinBox->setEnabled(true);
            p_sbMaxBox->setEnabled(true);
        }
        else
        {
            p_mainPlotBox->setItemEnabled(Both_SideBands,false);
            p_mainPlotBox->setItemEnabled(Upper_SideBand,false);
            p_mainPlotBox->setItemEnabled(Lower_SideBand,false);
            p_sbMinBox->setEnabled(false);
            p_sbMaxBox->setEnabled(false);
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

        for(auto it = d_seg.begin(); it != d_seg.end(); ++it)
        {
            it->second->blockSignals(true);
            it->second->setRange(1,e.ftmwConfig()->d_rfConfig.numSegments());
            it->second->blockSignals(false);
        }
        for(auto it = d_frame.begin(); it != d_frame.end(); ++it)
        {
            it->second->blockSignals(true);
            it->second->setRange(1,e.ftmwConfig()->d_scopeConfig.d_numRecords);
            it->second->blockSignals(false);
        }
        for(auto it = d_seg.begin(); it != d_seg.end(); ++it)
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

FtmwPlotToolBar::MainPlotMode FtmwPlotToolBar::mainPlotMode() const
{
    return p_mainPlotBox->value();
}

int FtmwPlotToolBar::mainPlotFollow() const
{
    return p_followBox->value();
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
