#include <gui/widget/ftmwplotconfigwidget.h>

#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGroupBox>
#include <QFormLayout>
#include <QSpinBox>
#include <QLabel>
#include <QPushButton>
#include <QRadioButton>
#include <QCheckBox>
#include <QListWidgetItem>
#include <QFile>
#include <QThread>
#include <QMessageBox>

FtmwPlotConfigWidget::FtmwPlotConfigWidget(int id, QWidget *parent) :
    QWidget(parent), d_id(id)
{
    auto vbl = new QVBoxLayout;

    auto fl = new QFormLayout;

    p_frameBox = new QSpinBox;
    p_frameBox->setRange(1,1);
    p_frameBox->setEnabled(false);
    p_frameBox->setKeyboardTracking(false);
    connect(p_frameBox,static_cast<void (QSpinBox::*)(int)>(&QSpinBox::valueChanged),this,[=](int v){
        emit frameChanged(d_id,v-1);
    });

    auto flbl = new QLabel("Frame Number");
    flbl->setSizePolicy(QSizePolicy::Minimum,QSizePolicy::MinimumExpanding);
    fl->addRow(flbl,p_frameBox);

    p_segmentBox = new QSpinBox;
    p_segmentBox->setRange(1,1);
    p_segmentBox->setEnabled(false);
    p_segmentBox->setKeyboardTracking(false);
    connect(p_segmentBox,static_cast<void (QSpinBox::*)(int)>(&QSpinBox::valueChanged),this,[=](int v){
        emit segmentChanged(d_id,v-1);
    });

    auto slbl = new QLabel("Segment Number");
    slbl->setSizePolicy(QSizePolicy::Minimum,QSizePolicy::MinimumExpanding);
    fl->addRow(slbl,p_segmentBox);

    p_autosaveBox = new QSpinBox;
    p_autosaveBox->setRange(0,0);
    p_autosaveBox->setEnabled(false);
    p_autosaveBox->setSpecialValueText("All");

    vbl->addLayout(fl);

    setLayout(vbl);

}

FtmwPlotConfigWidget::~FtmwPlotConfigWidget()
{
}

void FtmwPlotConfigWidget::prepareForExperiment(const Experiment &e)
{
    blockSignals(true);

    //these things only become enabled once snapshots have been taken
    p_autosaveBox->setRange(0,0);
    p_autosaveBox->setEnabled(false);

    if(e.ftmwEnabled())
    {
        p_frameBox->setRange(1,e.ftmwConfig()->d_scopeConfig.d_numRecords);
        p_frameBox->setEnabled(true);

        p_segmentBox->setRange(1,e.ftmwConfig()->d_rfConfig.numSegments());
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

void FtmwPlotConfigWidget::newAutosave(int numAutosaves)
{
    p_autosaveBox->setMaximum(numAutosaves);
    p_autosaveBox->setEnabled(true);
}

bool FtmwPlotConfigWidget::viewingAutosave()
{
    return p_autosaveBox->value()>0;
}

