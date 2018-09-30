#include "ftmwplotconfigwidget.h"

#include <QFormLayout>
#include <QSpinBox>
#include <QLabel>

FtmwPlotConfigWidget::FtmwPlotConfigWidget(QWidget *parent) : QWidget(parent)
{
    auto fl = new QFormLayout;

    p_frameBox = new QSpinBox;
    p_frameBox->setRange(1,1);
    p_frameBox->setEnabled(false);
    p_frameBox->setKeyboardTracking(false);
    connect(p_frameBox,static_cast<void (QSpinBox::*)(int)>(&QSpinBox::valueChanged),this,[=](int v){
        emit frameChanged(v-1);
    });

    auto flbl = new QLabel("Frame Number");
    flbl->setSizePolicy(QSizePolicy::Minimum,QSizePolicy::MinimumExpanding);
    fl->addRow(flbl,p_frameBox);

    p_segmentBox = new QSpinBox;
    p_segmentBox->setRange(1,1);
    p_segmentBox->setEnabled(false);
    p_segmentBox->setKeyboardTracking(false);
    connect(p_segmentBox,static_cast<void (QSpinBox::*)(int)>(&QSpinBox::valueChanged),this,[=](int v){
        emit segmentChanged(v-1);
    });

    auto slbl = new QLabel("Segment Number");
    slbl->setSizePolicy(QSizePolicy::Minimum,QSizePolicy::MinimumExpanding);
    fl->addRow(slbl,p_segmentBox);


    setLayout(fl);
}

void FtmwPlotConfigWidget::prepareForExperiment(const Experiment e)
{
    blockSignals(true);

    if(e.ftmwConfig().isEnabled())
    {
        p_frameBox->setRange(1,e.ftmwConfig().numFrames());
        p_frameBox->setEnabled(true);

        p_segmentBox->setRange(1,e.ftmwConfig().numSegments());
        p_segmentBox->setEnabled(true);
    }
    else
    {
        p_frameBox->setRange(1,1);
        p_frameBox->setEnabled(false);

        p_segmentBox->setRange(1,1);
        p_segmentBox->setEnabled(false);
    }

    blockSignals(false);
}
