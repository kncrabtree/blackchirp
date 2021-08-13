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

FtmwPlotConfigWidget::FtmwPlotConfigWidget(QWidget *parent) :
    QWidget(parent)
{
    auto vbl = new QVBoxLayout;

    auto fl = new QFormLayout;

    p_frameBox = new QSpinBox;
    p_frameBox->setRange(1,1);
    p_frameBox->setEnabled(false);
    p_frameBox->setKeyboardTracking(false);
    connect(p_frameBox,qOverload<int>(&QSpinBox::valueChanged),[this](int v){
        emit frameChanged(v-1);
    });

    auto flbl = new QLabel("Frame Number");
    flbl->setSizePolicy(QSizePolicy::Minimum,QSizePolicy::MinimumExpanding);
    fl->addRow(flbl,p_frameBox);

    p_segmentBox = new QSpinBox;
    p_segmentBox->setRange(1,1);
    p_segmentBox->setEnabled(false);
    p_segmentBox->setKeyboardTracking(false);
    connect(p_segmentBox,qOverload<int>(&QSpinBox::valueChanged),[this](int v){
        emit segmentChanged(v-1);
    });

    auto slbl = new QLabel("Segment Number");
    slbl->setSizePolicy(QSizePolicy::Minimum,QSizePolicy::MinimumExpanding);
    fl->addRow(slbl,p_segmentBox);

    p_backupBox = new QSpinBox;
    p_backupBox->setRange(0,0);
    p_backupBox->setEnabled(false);
    p_backupBox->setSpecialValueText("All");

    auto blbl = new QLabel("Backup Number");
    blbl->setSizePolicy(QSizePolicy::Minimum,QSizePolicy::MinimumExpanding);
    fl->addRow(blbl,p_backupBox);
    connect(p_backupBox,qOverload<int>(&QSpinBox::valueChanged),this,&FtmwPlotConfigWidget::backupChanged);

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
    p_backupBox->setRange(0,0);
    p_backupBox->setEnabled(false);

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

void FtmwPlotConfigWidget::newBackup(int numBackups)
{
    p_backupBox->setMaximum(numBackups);
    p_backupBox->setEnabled(true);
}

bool FtmwPlotConfigWidget::viewingBackup()
{
    return p_backupBox->value()>0;
}

