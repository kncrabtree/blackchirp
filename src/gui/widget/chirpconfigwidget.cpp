#include "chirpconfigwidget.h"
#include "ui_chirpconfigwidget.h"

#include <QMessageBox>
#include <QSpinBox>
#include <QInputDialog>

#include <hardware/core/ftmwdigitizer/ftmwscope.h>
#include <hardware/optional/chirpsource/awg.h>

ChirpConfigWidget::ChirpConfigWidget(QWidget *parent) :
    QWidget(parent), SettingsStorage(BC::Key::ChirpConfigWidget::key),
    ui(new Ui::ChirpConfigWidget), p_ctm(new ChirpTableModel(this)), d_rampOnly(false)
{
    ui->setupUi(this);
    ui->chirpTable->setModel(p_ctm);
    ui->chirpTable->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);

    SettingsStorage s(BC::Key::FtmwScope::ftmwScope,SettingsStorage::Hardware);
    bool ff = s.get(BC::Key::Digi::canMultiRecord,false);

    if(!ff)
    {
        ui->chirpsSpinBox->setValue(1);
        ui->chirpsSpinBox->setEnabled(false);
    }

    SettingsStorage awg(BC::Key::AWG::key,SettingsStorage::Hardware);
    d_awgSampleRate = awg.get(BC::Key::AWG::rate,16e9);

    using namespace BC::Key::ChirpConfigWidget;
    QString us = QString::fromUtf8(" μs");
    double _minPreProt = getOrSetDefault(minPreProt,0.0);
    double _minPreGate = getOrSetDefault(minPreGate,0.0);
    double _minPostGate = getOrSetDefault(minPostGate,-0.5);
    double _minPostProt = getOrSetDefault(minPostProt,0.0);

    ui->preChirpProtectionDoubleSpinBox->setMinimum(_minPreProt);
    ui->preChirpDelayDoubleSpinBox->setMinimum(_minPreGate);
    ui->postChirpDelayDoubleSpinBox->setMinimum(_minPostGate);
    ui->postChirpProtectionDoubleSpinBox->setMinimum(_minPostProt);

    ui->preChirpDelayDoubleSpinBox->setSuffix(us);
    ui->preChirpProtectionDoubleSpinBox->setSuffix(us);
    ui->postChirpDelayDoubleSpinBox->setSuffix(us);
    ui->postChirpProtectionDoubleSpinBox->setSuffix(us);
    ui->chirpIntervalDoubleSpinBox->setSuffix(us);

    ui->preChirpProtectionDoubleSpinBox->setValue(get(preProt,0.5));
    ui->postChirpProtectionDoubleSpinBox->setValue(get(postProt,0.5));
    ui->preChirpDelayDoubleSpinBox->setValue(get(preGate,0.5));
    ui->postChirpDelayDoubleSpinBox->setValue(get(postGate,0.5));
    ui->chirpsSpinBox->setValue(get(numChirps,1));
    ui->chirpIntervalDoubleSpinBox->setValue(get(interval,20));
    ui->chirpIntervalDoubleSpinBox->setEnabled(ui->chirpsSpinBox->value()>1);
    ui->applyToAllBox->setChecked(get(applyAll,true));

    registerGetter(preProt,ui->preChirpProtectionDoubleSpinBox,&QDoubleSpinBox::value);
    registerGetter(postProt,ui->postChirpProtectionDoubleSpinBox,&QDoubleSpinBox::value);
    registerGetter(preGate,ui->preChirpDelayDoubleSpinBox,&QDoubleSpinBox::value);
    registerGetter(postGate,ui->postChirpDelayDoubleSpinBox,&QDoubleSpinBox::value);
    registerGetter(numChirps,ui->chirpsSpinBox,&QSpinBox::value);
    registerGetter(interval,ui->chirpIntervalDoubleSpinBox,&QDoubleSpinBox::value);
    registerGetter<QAbstractButton>(applyAll,ui->applyToAllBox,&QCheckBox::isChecked);



    connect(p_ctm,&ChirpTableModel::modelChanged,this,&ChirpConfigWidget::setButtonStates);
    connect(ui->chirpTable->selectionModel(),&QItemSelectionModel::selectionChanged,this,&ChirpConfigWidget::setButtonStates);
    connect(p_ctm,&ChirpTableModel::modelChanged,this,&ChirpConfigWidget::updateChirpPlot);

    setButtonStates();



    connect(ui->addButton,&QPushButton::clicked,this,&ChirpConfigWidget::addSegment);
    connect(ui->addEmptyButton,&QPushButton::clicked,this,&ChirpConfigWidget::addEmptySegment);
    connect(ui->insertButton,&QPushButton::clicked,this,&ChirpConfigWidget::insertSegment);
    connect(ui->insertEmptyButton,&QPushButton::clicked,this,&ChirpConfigWidget::insertEmptySegment);
    connect(ui->moveUpButton,&QPushButton::clicked,[=](){ moveSegments(-1); });
    connect(ui->moveDownButton,&QPushButton::clicked,[=](){ moveSegments(1); });
    connect(ui->removeButton,&QPushButton::clicked,this,&ChirpConfigWidget::removeSegments);
    connect(ui->clearButton,&QPushButton::clicked,this,&ChirpConfigWidget::clear);

    auto vc = static_cast<void (QSpinBox::*)(int)>(&QSpinBox::valueChanged);
    auto dvc = static_cast<void (QDoubleSpinBox::*)(double)>(&QDoubleSpinBox::valueChanged);
    connect(ui->preChirpProtectionDoubleSpinBox,dvc,this,&ChirpConfigWidget::updateChirpPlot);
    connect(ui->preChirpDelayDoubleSpinBox,dvc,this,&ChirpConfigWidget::updateChirpPlot);
    connect(ui->postChirpDelayDoubleSpinBox,dvc,this,&ChirpConfigWidget::updateChirpPlot);
    connect(ui->postChirpProtectionDoubleSpinBox,dvc,this,&ChirpConfigWidget::updateChirpPlot);
    connect(ui->chirpsSpinBox,vc,p_ctm,&ChirpTableModel::setNumChirps);
    connect(ui->chirpsSpinBox,vc,ui->currentChirpBox,&QSpinBox::setMaximum);
    connect(ui->chirpsSpinBox,vc,this,&ChirpConfigWidget::updateChirpPlot);
    connect(ui->chirpsSpinBox,vc,[=](int n){ui->chirpIntervalDoubleSpinBox->setEnabled(n>1);});
    connect(ui->chirpsSpinBox,vc,this,&ChirpConfigWidget::setButtonStates);
    connect(ui->chirpIntervalDoubleSpinBox,dvc,this,&ChirpConfigWidget::updateChirpPlot);
    connect(ui->currentChirpBox,vc,[=](int val){ p_ctm->setCurrentChirp(val-1); });
    connect(ui->applyToAllBox,&QCheckBox::toggled,[this](bool en){p_ctm->d_allIdentical = en;});

    ui->chirpTable->setItemDelegate(new ChirpDoubleSpinBoxDelegate);


}

ChirpConfigWidget::~ChirpConfigWidget()
{
    clearGetters(false);
    delete ui;
}

void ChirpConfigWidget::initialize(RfConfig *p)
{
    p_rfConfig = p;
    p_ctm->initialize(p);
    updateChirpPlot();
}

void ChirpConfigWidget::setFromRfConfig(RfConfig *p)
{
    disconnect(p_ctm,&ChirpTableModel::modelChanged,this,&ChirpConfigWidget::updateChirpPlot);
    p_rfConfig = p;
    if(d_rampOnly)
    {
        auto thiscc = p_rfConfig->d_chirpConfig;
        if(!thiscc.chirpList().isEmpty())
        {
            if(thiscc.chirpList().constFirst().size() > 1)
            {
                thiscc.setChirpList(QVector<QVector<ChirpConfig::ChirpSegment>>());
                p_rfConfig->setChirpConfig(thiscc);
            }
        }
    }



    auto &cc = p_rfConfig->d_chirpConfig;

    ui->preChirpProtectionDoubleSpinBox->blockSignals(true);
    ui->preChirpProtectionDoubleSpinBox->setValue(cc.preChirpProtectionDelay());
    ui->preChirpProtectionDoubleSpinBox->blockSignals(false);

    ui->preChirpDelayDoubleSpinBox->blockSignals(true);
    ui->preChirpDelayDoubleSpinBox->setValue(cc.preChirpGateDelay());
    ui->preChirpDelayDoubleSpinBox->blockSignals(false);

    ui->postChirpDelayDoubleSpinBox->blockSignals(true);
    ui->postChirpDelayDoubleSpinBox->setValue(cc.postChirpGateDelay());
    ui->postChirpDelayDoubleSpinBox->blockSignals(false);

    ui->postChirpProtectionDoubleSpinBox->blockSignals(true);
    ui->postChirpProtectionDoubleSpinBox->setValue(cc.postChirpProtectionDelay());
    ui->postChirpProtectionDoubleSpinBox->blockSignals(false);

    ui->chirpsSpinBox->blockSignals(true);
    ui->chirpsSpinBox->setValue(cc.numChirps());
    ui->chirpsSpinBox->blockSignals(false);

    ui->applyToAllBox->blockSignals(true);
    ui->applyToAllBox->setChecked(cc.allChirpsIdentical());
    ui->applyToAllBox->blockSignals(false);


    if(!cc.chirpList().isEmpty())
    {
        ui->chirpIntervalDoubleSpinBox->blockSignals(true);
        ui->chirpIntervalDoubleSpinBox->setValue(cc.chirpInterval());
        ui->chirpIntervalDoubleSpinBox->blockSignals(false);
    }

    ui->currentChirpBox->blockSignals(true);
    ui->currentChirpBox->setValue(1);
    ui->currentChirpBox->blockSignals(false);

    p_ctm->setFromRfConfig(p);
    p_ctm->d_allIdentical = cc.allChirpsIdentical();
    connect(p_ctm,&ChirpTableModel::modelChanged,this,&ChirpConfigWidget::updateChirpPlot);
    updateChirpPlot();
}

void ChirpConfigWidget::enableEditing(bool enabled)
{
    ui->chirpConfigurationBox->setEnabled(enabled);
}

void ChirpConfigWidget::setButtonStates()
{
    //set which buttons should be enabled
    //get selection
    QModelIndexList l = ui->chirpTable->selectionModel()->selectedRows();
    bool c = isSelectionContiguous(l);

    if(d_rampOnly)
    {
        ui->addButton->setEnabled(p_ctm->rowCount(QModelIndex()) == 0);
        ui->addEmptyButton->setEnabled(false);
    }

    //insert button only enabled if one item is selected
    ui->insertButton->setEnabled(l.size() == 1 && !d_rampOnly);
    ui->insertEmptyButton->setEnabled(l.size() == 1 && !d_rampOnly);

    //remove button active if one or more rows selected
    ui->removeButton->setEnabled(l.size() > 0);

    //move buttons enabled only if selection is contiguous
    ui->moveDownButton->setEnabled(c && p_ctm->rowCount(QModelIndex()) > 0);
    ui->moveUpButton->setEnabled(c && p_ctm->rowCount(QModelIndex()) > 0);

    //clear button only enabled if table is not empty
    ui->clearButton->setEnabled(p_ctm->rowCount(QModelIndex()) > 0);

    //get number of chirps associated with current chirp config
    auto cl = p_ctm->chirpList();
    ui->currentChirpBox->setRange(1,qMax(1,cl.size()));
    ui->applyToAllBox->setEnabled(p_ctm->d_allIdentical);
}

void ChirpConfigWidget::addSegment()
{
    p_ctm->addSegment(-1.0,-1.0,0.500,-1);
    updateChirpPlot();
}

void ChirpConfigWidget::addEmptySegment()
{
    p_ctm->addSegment(0.0,0.0,0.500,-1,true);
    updateChirpPlot();
}

void ChirpConfigWidget::insertSegment()
{
    QModelIndexList l = ui->chirpTable->selectionModel()->selectedRows();
    if(l.size() != 1)
        return;

    p_ctm->addSegment(-1.0,-1.0,0.500,l.at(0).row());
    updateChirpPlot();
}

void ChirpConfigWidget::insertEmptySegment()
{
    QModelIndexList l = ui->chirpTable->selectionModel()->selectedRows();
    if(l.size() != 1)
        return;

    p_ctm->addSegment(0.0,0.0,0.500,l.at(0).row(),true);
    updateChirpPlot();
}

void ChirpConfigWidget::moveSegments(int direction)
{
    QModelIndexList l = ui->chirpTable->selectionModel()->selectedRows();

    if(l.isEmpty())
        return;

    //get the selected rows
    QList<int> sortList;
    for(int i=0; i<l.size(); i++)
        sortList.append(l.at(i).row());

    std::sort(sortList.begin(),sortList.end());

    //make sure selection is contiguous
    if(sortList.size()>1 && sortList.at(sortList.size()-1) - sortList.at(0) != sortList.size()-1)
        return;

    p_ctm->moveSegments(sortList.at(0),sortList.at(sortList.size()-1),direction);
    updateChirpPlot();
}

void ChirpConfigWidget::removeSegments()
{
    QModelIndexList l = ui->chirpTable->selectionModel()->selectedRows();

    if(l.isEmpty())
        return;

    QList<int> rows;
    for(int i=0;i<l.size();i++)
        rows.append(l.at(i).row());

    p_ctm->removeSegments(rows);
    updateChirpPlot();
}

void ChirpConfigWidget::clear()
{
    int ret = QMessageBox::question(this,QString("Clear all rows"),
                          QString("Are you sure you want to remove all segments from the table?"),
                          QMessageBox::Yes|QMessageBox::Cancel, QMessageBox::Cancel);

    if(ret == QMessageBox::Yes)
        clearList();
}

void ChirpConfigWidget::updateChirpPlot()
{
    updateRfConfig();

    emit chirpConfigChanged();
    ui->chirpPlot->newChirp(p_rfConfig->d_chirpConfig);
}

void ChirpConfigWidget::updateRfConfig()
{
    auto l = p_ctm->chirpList();
    auto &cc = p_rfConfig->d_chirpConfig;
    cc.setPreChirpProtectionDelay(ui->preChirpProtectionDoubleSpinBox->value());
    cc.setPreChirpGateDelay(ui->preChirpDelayDoubleSpinBox->value());
    cc.setPostChirpGateDelay(ui->postChirpDelayDoubleSpinBox->value());
    cc.setPostChirpProtectionDelay(ui->postChirpProtectionDoubleSpinBox->value());
    cc.setNumChirps(ui->chirpsSpinBox->value());
    cc.setChirpInterval(ui->chirpIntervalDoubleSpinBox->value());
    cc.setChirpList(l);
    cc.setAwgSampleRate(d_awgSampleRate);
}

bool ChirpConfigWidget::isSelectionContiguous(QModelIndexList l)
{
    if(l.isEmpty())
        return false;

    if(l.size()==1)
        return true;

    //selection is contiguous if the last row minus first row is equal to the size, after the list has been sorted
    QList<int> sortList;
    for(int i=0; i<l.size(); i++)
        sortList.append(l.at(i).row());

    std::sort(sortList.begin(),sortList.end());

    if(sortList.at(sortList.size()-1) - sortList.at(0) != sortList.size()-1)
        return false;

    return true;
}

void ChirpConfigWidget::clearList(bool replot)
{
    if(p_ctm->rowCount(QModelIndex()) > 0)
        p_ctm->removeRows(0,p_ctm->rowCount(QModelIndex()),QModelIndex());

    if(replot)
        updateChirpPlot();
}
